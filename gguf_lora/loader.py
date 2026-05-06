import torch
import torch.nn as nn
from gguf import GGUFReader, GGMLQuantizationType
from gguf_lora.naming import get_name_map_from_reader
from gguf_lora.quant import dequantize, is_supported_quant_type, normalise_quant_type
from gguf_lora.lora import LazyGGUFLinear

SUPPORTED_QUANT_TYPES = {"Q8_0", GGMLQuantizationType.Q8_0, 8}

class GGUFLoader:
    def __init__(self, gguf_path):
        self.reader = GGUFReader(gguf_path)
        self.name_map = get_name_map_from_reader(self.reader)
        self.tensors = {}
        self.modules = {}
        self._parse_tensors()
        self._build_modules()

    def _parse_tensors(self):
        for tensor in self.reader.tensors:
            hf_name = self.name_map.gguf_to_hf(tensor.name)
            if hf_name is None:
                continue
            raw_data = bytes(tensor.data.tobytes())
            # Convert shape to plain Python ints
            shape = tuple(int(s) for s in tensor.shape)
            if tensor.name == "token_embd.weight" or hf_name == "model.embed_tokens.weight":
                print(f"[DEBUG] GGUF tensor '{tensor.name}' raw shape: {tensor.shape}")
            self.tensors[hf_name] = {
                'gguf_name': tensor.name,
                'type': tensor.tensor_type,
                'shape': shape,
                'raw_data': raw_data,
            }

    def _build_module(self, hf_name, meta):
        quant_type = meta['type']
        shape = meta['shape']
        raw_data = meta['raw_data']
        # Always load embedding weights as float32 tensor
        if hf_name.endswith("embed_tokens.weight"):
            print(f"[DEBUG] Building module for {hf_name} with shape {shape} and quant_type {quant_type}")
            # Swap shape if needed: GGUF stores as [vocab_size, hidden_dim], but if shape[0] < shape[1], swap
            if shape[0] < shape[1]:
                print(f"[DEBUG] Swapping embedding shape from {shape} to {(shape[1], shape[0])}")
                shape = (shape[1], shape[0])
            # If quantized, dequantize first
            if quant_type == GGMLQuantizationType.F32:
                data = torch.frombuffer(bytearray(raw_data), dtype=torch.float32).reshape(shape)
            else:
                data = dequantize(raw_data, normalise_quant_type(quant_type), shape)
            print(f"[DEBUG] Final embedding parameter shape: {data.shape}")
            return nn.Parameter(data, requires_grad=False)
        # For linear/projection weights, always handle GGUF convention
        if any(s in hf_name for s in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"]):
            if quant_type == GGMLQuantizationType.F32:
                tensor = torch.frombuffer(bytearray(raw_data), dtype=torch.float32).reshape(shape)
                tensor = tensor.T  # GGUF is always [in, out], PyTorch wants [out, in]
                print(f"[DEBUG] Final projection parameter shape for {hf_name}: {tensor.shape}")
                return nn.Parameter(tensor.contiguous(), requires_grad=False)
            else:
                # For quantized, pass raw_data and shape; transpose in dequantize
                return LazyGGUFLinear(
                    quantized_data=raw_data,
                    quant_type=normalise_quant_type(quant_type),
                    shape=shape
                )
        if quant_type == GGMLQuantizationType.F32:
            data = torch.frombuffer(bytearray(raw_data), dtype=torch.float32).reshape(shape)
            return nn.Parameter(data, requires_grad=False)
        elif is_supported_quant_type(quant_type):
            return LazyGGUFLinear(
                quantized_data=raw_data,
                quant_type=normalise_quant_type(quant_type),
                shape=shape
            )
        else:
            raise NotImplementedError(
                f"Tensor '{hf_name}' has unsupported quant type {quant_type}. "
                f"See quant.py for supported types."
            )

    def _build_modules(self):
        for hf_name, meta in self.tensors.items():
            self.modules[hf_name] = self._build_module(hf_name, meta)

    def get_module(self, hf_name):
        return self.modules.get(hf_name)

    def list_hf_tensors(self):
        return list(self.tensors.keys())

    def get_qwen3_config(self):
        fields = self.reader.fields
        # Cleaned up: remove debug prints for production use
        # print("[DEBUG] GGUF relevant fields (raw):")
        # for k, v in fields.items():
        #     k_str = k.decode() if isinstance(k, bytes) else k
        #     if any(x in k_str for x in ["head", "length", "dim", "block", "hidden", "embed", "layer", "context", "rope"]):
        #         print(f"  {k_str}: type={type(v).__name__}, value={v}, data={getattr(v, 'data', None)}")
        # print("[DEBUG] GGUF relevant fields:")
        # for k, v in fields.items():
        #     k_str = k.decode() if isinstance(k, bytes) else k
        #     if any(x in k_str for x in ["head", "length", "dim", "block", "hidden", "embed", "layer", "context", "rope"]):
        #         try:
        #             val = getattr(v, 'data', v)
        #             if hasattr(val, '__len__') and len(val) > 10:
        #                 print(f"  {k_str}: <{len(val)} elements, type {type(val).__name__}>")
        #             else:
        #                 print(f"  {k_str}: {val}")
        #         except Exception as e:
        #             print(f"  {k_str}: <unprintable> {e}")
        # Build a string-keyed fields dict for robust lookup
        str_fields = {k.decode() if isinstance(k, bytes) else k: v for k, v in fields.items()}
        def get_field(key, default=None):
            field = str_fields.get(key, None)
            if field is not None:
                # Try to extract from last part (should be a memmap with the value)
                try:
                    val = field.parts[-1][0]
                    return int(val)
                except Exception:
                    pass
                # Fallback to .data[0]
                try:
                    return int(field.data[0])
                except Exception:
                    pass
            return default
        return {
            "hidden_size": get_field("qwen3.embedding_length"),
            "num_hidden_layers": get_field("qwen3.block_count"),
            "num_attention_heads": get_field("qwen3.attention.head_count"),
            "num_key_value_heads": get_field("qwen3.attention.head_count_kv"),
            "head_dim": get_field("qwen3.attention.key_length"),
            "intermediate_size": get_field("qwen3.feed_forward_length"),
            "rope_theta": get_field("qwen3.rope.freq_base", 1000000.0),
            "max_position_embeddings": get_field("qwen3.context_length", 32768),
        }

# Example usage (remove or adapt for tests/CLI):
if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    loader = GGUFLoader(config["model_path"])
    print("Mapped HF tensor names:", loader.list_hf_tensors())
