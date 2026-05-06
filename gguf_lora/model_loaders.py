# model_loaders.py
# Central place to import all model loader classes (Qwen3, Llama, etc.)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Qwen3-specific RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

# --- RoPE utilities ---
def precompute_rope(dim, max_seq_len, base=1000000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin

def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

# --- Qwen3 Attention ---
class Qwen3Attention(nn.Module):
    def __init__(self, modules, layer_idx, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.layer_idx = layer_idx
        prefix = f"model.layers.{layer_idx}.self_attn"
        self.q_proj = modules[f"{prefix}.q_proj.weight"]
        self.k_proj = modules[f"{prefix}.k_proj.weight"]
        self.v_proj = modules[f"{prefix}.v_proj.weight"]
        self.o_proj = modules[f"{prefix}.o_proj.weight"]
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _proj(self, x, proj, out_features=None):
        # If proj is a module (e.g., LazyGGUFLinear), call it; if tensor/Parameter, use F.linear
        if hasattr(proj, 'forward'):
            return proj(x)
        else:
            # out_features is needed for correct shape if proj is a weight tensor
            return F.linear(x, proj)

    def forward(self, x, cos, sin, position_ids, attention_mask=None):
        B, T, _ = x.shape
        # Compute expected last dim for projections
        q_out = self.num_heads * self.head_dim
        k_out = self.num_kv_heads * self.head_dim
        v_out = self.num_kv_heads * self.head_dim
        # Projections
        q = self._proj(x, self.q_proj)
        k = self._proj(x, self.k_proj)
        v = self._proj(x, self.v_proj)
        # Check shapes and reshape
        assert q.shape[-1] == q_out, f"q_proj output {q.shape[-1]} != num_heads*head_dim {q_out}"
        assert k.shape[-1] == k_out, f"k_proj output {k.shape[-1]} != num_kv_heads*head_dim {k_out}"
        assert v.shape[-1] == v_out, f"v_proj output {v.shape[-1]} != num_kv_heads*head_dim {v_out}"
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin, position_ids)
        if self.num_kv_heads != self.num_heads:
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self._proj(attn_out, self.o_proj)

# --- Qwen3 MLP ---
class Qwen3MLP(nn.Module):
    def __init__(self, modules, layer_idx):
        super().__init__()
        prefix = f"model.layers.{layer_idx}.mlp"
        self.gate_proj = modules[f"{prefix}.gate_proj.weight"]
        self.up_proj = modules[f"{prefix}.up_proj.weight"]
        self.down_proj = modules[f"{prefix}.down_proj.weight"]

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# --- Qwen3 Decoder Layer ---
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, modules, layer_idx, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.attn = Qwen3Attention(modules, layer_idx, config)
        self.mlp = Qwen3MLP(modules, layer_idx)
        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.input_layernorm.weight = modules[f"{prefix}.input_layernorm.weight"]
        self.post_attention_layernorm.weight = modules[f"{prefix}.post_feedforward_layernorm.weight"]

    def forward(self, x, cos, sin, position_ids, attention_mask=None):
        residual = x
        x = self.attn(self.input_layernorm(x), cos, sin, position_ids, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        return residual + x

# --- Qwen3ForCausalLM ---
class Qwen3ForCausalLM(nn.Module):
    """
    Full Qwen3 model for GGUF-LoRA: embedding, transformer stack, output projection.
    Assumes modules dict (with LoRA injected) from GGUFLoader and a config dict.
    """
    def __init__(self, modules, config):
        super().__init__()
        self.config = config
        self.embedding = modules["model.embed_tokens.weight"]
        self.norm = RMSNorm(config["hidden_size"])
        self.norm.weight = modules["model.norm.weight"]
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(modules, i, config)
            for i in range(config["num_hidden_layers"])
        ])
        self.lm_head = modules["model.embed_tokens.weight"]
        cos, sin = precompute_rope(
            config["head_dim"],
            max_seq_len=config.get("max_position_embeddings", 32768),
            base=config.get("rope_theta", 1000000.0)
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, input_ids, attention_mask=None):
        # Ensure input_ids is always 2D (batch, seq_len)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = F.embedding(input_ids, self.embedding)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, position_ids, attention_mask)
        x = self.norm(x)
        logits = F.linear(x, self.embedding)
        return logits

__all__ = [
    "Qwen3ForCausalLM",
    # "LlamaForCausalLM",
]
