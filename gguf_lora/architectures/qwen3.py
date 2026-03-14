ARCHITECTURE_ID = "qwen3"  # matches GGUF metadata string

# Tensors that are Q8_0 and are valid LoRA targets
QUANTIZED_TENSORS = {
    "blk.{i}.attn_q.weight":      "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight":      "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight":      "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.ffn_gate.weight":    "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight":      "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight":    "model.layers.{i}.mlp.down_proj.weight",
}

# Tensors that are F32 — loaded as plain nn.Parameters, never LoRA targets
UNQUANTIZED_TENSORS = {
    "blk.{i}.attn_norm.weight":   "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.ffn_norm.weight":    "model.layers.{i}.post_feedforward_layernorm.weight",
    "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
    "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
}

# Top-level tensors (outside blocks)
TOP_LEVEL = {
    "output_norm.weight": "model.norm.weight",           # F32
    "token_embd.weight":  "model.embed_tokens.weight",   # Q8_0 but not a LoRA target
}

# Default LoRA injection targets for this architecture
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj",                   # FFN (down_proj optional, adds params)
]

NUM_LAYERS = 36

class Qwen3NameMap:
    ARCHITECTURE_ID = ARCHITECTURE_ID
    NUM_LAYERS = NUM_LAYERS
    QUANTIZED_TENSORS = QUANTIZED_TENSORS
    UNQUANTIZED_TENSORS = UNQUANTIZED_TENSORS
    TOP_LEVEL = TOP_LEVEL
    DEFAULT_TARGET_MODULES = DEFAULT_TARGET_MODULES

    def gguf_to_hf(self, gguf_name: str) -> str:
        for i in range(self.NUM_LAYERS):
            for gguf_pat, hf_pat in self.QUANTIZED_TENSORS.items():
                if gguf_pat.format(i=i) == gguf_name:
                    return hf_pat.format(i=i)
            for gguf_pat, hf_pat in self.UNQUANTIZED_TENSORS.items():
                if gguf_pat.format(i=i) == gguf_name:
                    return hf_pat.format(i=i)
        for gguf_pat, hf_pat in self.TOP_LEVEL.items():
            if gguf_pat == gguf_name:
                return hf_pat
        return None

    def hf_to_gguf(self, hf_name: str) -> str | None:
        # Reverse mapping for quantized and unquantized tensors
        for i in range(self.NUM_LAYERS):
            for gguf_pat, hf_pat in self.QUANTIZED_TENSORS.items():
                if hf_pat.format(i=i) == hf_name:
                    return gguf_pat.format(i=i)
            for gguf_pat, hf_pat in self.UNQUANTIZED_TENSORS.items():
                if hf_pat.format(i=i) == hf_name:
                    return gguf_pat.format(i=i)
        for gguf_pat, hf_pat in self.TOP_LEVEL.items():
            if hf_pat == hf_name:
                return gguf_pat
        return None
