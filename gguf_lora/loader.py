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
        # Accept both enum, int, and string for Q8_0
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

# Example usage (remove or adapt for tests/CLI):
if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    loader = GGUFLoader(config["model_path"])
    print("Mapped HF tensor names:", loader.list_hf_tensors())
