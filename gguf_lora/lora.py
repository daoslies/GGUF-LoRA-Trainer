import torch
import torch.nn as nn
from gguf import GGMLQuantizationType
from gguf_lora.quant import dequantize

__all__ = ["LazyGGUFLinear", "LoRAGGUFLinear", "inject_lora"]

class LazyGGUFLinear(nn.Module):
    def __init__(self, quantized_data, quant_type, shape):
        super().__init__()
        self.quantized_data = quantized_data
        self.quant_type = quant_type
        self.shape = shape
    def forward(self, x):
        weight = dequantize(self.quantized_data, self.quant_type, self.shape)
        return nn.functional.linear(x, weight)

class LoRAGGUFLinear(LazyGGUFLinear):
    def __init__(self, quantized_data, quant_type, shape, rank, alpha):
        super().__init__(quantized_data, quant_type, shape)
        in_features = shape[1]
        out_features = shape[0]
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    def forward(self, x):
        base_out = super().forward(x)
        lora_out = nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return base_out + (self.alpha / self.rank) * lora_out

def inject_lora(modules, target_modules, rank, alpha):
    """
    Replace eligible LazyGGUFLinear modules with LoRAGGUFLinear in-place.
    modules: dict of {name: module}
    target_modules: list of substrings (e.g. ["q_proj", ...])
    """
    for name, mod in list(modules.items()):
        if isinstance(mod, LazyGGUFLinear) and any(t in name for t in target_modules):
            modules[name] = LoRAGGUFLinear(
                mod.quantized_data, mod.quant_type, mod.shape, rank, alpha
            )
    return modules
