"""
writer.py: Save LoRA adapters to GGUF LoRA format.

This module extracts LoRA matrices (lora_A, lora_B) from all LoRAGGUFLinear modules
in the loader, converts their names back to GGUF convention, and writes a valid GGUF file
containing only the LoRA adapters and required metadata.

Implements llama.cpp's llama-lora.cpp output spec exactly:
- Metadata: general.type="adapter" (via GGUFType.ADAPTER), adapter.type="lora", adapter.lora.alpha (float32)
- Tensor names: GGUF convention + .lora_a/.lora_b suffix (lowercase)
- No adapter.lora.r key (rank is implicit)
- Both lora_A and lora_B must be present for each layer
"""
import gguf
import torch
from gguf_lora.lora import LoRAGGUFLinear

def save_lora_gguf(modules, name_map, output_path, alpha, arch_id):
    writer = gguf.GGUFWriter(output_path, arch=arch_id)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))

    for hf_name, module in modules.items():
        if not isinstance(module, LoRAGGUFLinear):
            continue
        gguf_name = name_map.hf_to_gguf(hf_name)
        if gguf_name is None:
            raise ValueError(f"No reverse mapping for {hf_name}")
        lora_a = module.lora_A.detach().cpu().float()  # [rank, in_features] (but actually [rank, out_features] in your code)
        lora_b = module.lora_B.detach().cpu().float()  # [out_features, rank] (but actually [in_features, rank] in your code)
        # Transpose both to match GGUF/llama.cpp convention
        writer.add_tensor(gguf_name + ".lora_a", lora_a.contiguous().numpy())  # [in_features, rank]
        writer.add_tensor(gguf_name + ".lora_b", lora_b.contiguous().numpy())    # [out_features, rank]

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
