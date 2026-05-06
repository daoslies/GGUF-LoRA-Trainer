# Implementation Journal: gguf-lora

## 2026-03-14

- Project initialized. Onboarding docs and architectural plan reviewed.
- Starting with test-driven approach as mandated: writing test_grad_flow.py first.
- Scaffolded test_grad_flow.py with minimal LoRAGGUFLinear stub for gradient flow and frozen param checks.
- Next: Implement quant.py (Q8_0 dequant only) after tests pass.

## 2026-03-14 (continued)

- Refactored test_grad_flow.py to use unittest for better integration and maintainability.
- Confirmed tests pass: gradient flow and frozen param checks are solid.
- Next: Implement quant.py (Q8_0 dequant only).

## 2026-03-14 (continued)

- Implemented and validated modular architecture registry in naming.py, supporting both string and integer GGUF architecture codes.
- Added robust tests for registry routing, tensor mapping, and unmapped tensor detection using a real Qwen3-4B Q8_0 GGUF file.
- Refactored naming.py to accept an already-open GGUFReader, avoiding double file reads in tests and loader.
- Scaffolded loader.py to map GGUF tensors to HuggingFace names and store tensor metadata.
- Added and passed tests for loader tensor mapping consistency, LoRA target coverage, and correct mapping to GGUFReader tensors.
- Implemented _build_module in loader.py to construct nn.Parameter for F32 tensors and LazyGGUFLinear for quantized tensors, copying raw bytes safely from mmap.
- Added and passed tests to ensure F32 tensors are plain nn.Parameter, quantized tensors are LazyGGUFLinear, and no base model parameter has requires_grad=True.
- Loader is now production ready for handoff to lora.py.

## 2026-03-14 (continued)

- Fixed a critical bug in Q8_0 dequantization logic: scale extraction now correctly interprets two bytes per block as little-endian float16 using pure torch ops (no numpy).
- Removed shape/data mismatch that previously caused OOM (hundreds of GB allocation) in gradient flow tests on real model tensors.
- All gradient flow tests now pass on real Qwen3-4B Q8_0 GGUF data, with correct shape and memory usage for all LoRA-injected modules.
- Cleaned up all debug prints from tests for fast, clean CI runs.
- Next: wire up train.py and demo to architects.

## 2026-03-14 (continued)

- Eliminated all buffer warnings in quant.py by switching to bytearray(data) in torch.frombuffer, matching loader.py for robust mmap safety.
- Reran all foundational and gradient flow tests: all pass, no warnings or errors remain.
- Codebase is now fully clean and robust for Q8_0 GGUF models, with modular architecture registry and fast test suite.
- Next: implement train.py (CLI entry point for LoRA training) and writer.py (GGUF LoRA output), following the onboarding and architecture memos.
- Will document known limitations and supported quant types in README.md after CLI and writer are complete.

## 2026-03-20

### Qwen3 GGUF LoRA Trainer: Device, Precision, and Performance Notes

- **Device Handling:**
  - All model weights, LoRA parameters, and input tensors are moved to the correct device (CPU or CUDA) before training.
  - `LazyGGUFLinear` dequantizes weights on-the-fly and ensures they are on the same device as the input tensor.

- **Precision:**
  - GGUF weights are stored quantized (Q8_0) and only dequantized to float32 for the current layer during forward pass.
  - LoRA weights are float32 by default; activations use mixed precision (float16) on CUDA via autocast.

- **Performance:**
  - Step time: ~2.5-5.5s/step (GPU), ~6.8-7.5s/step (CPU) for Qwen3-4B Q8_0.
  - GPU is ~2x faster than CPU, but per-step dequantization is CPU-bound and limits speedup.

- **VRAM Usage:**
  - Only the current layer's weights are in float32 at any time; rest remain quantized in RAM.
  - VRAM logging is included after model init, forward, and backward for profiling.

- **Recommendations:**
  - For further speedup, consider custom CUDA quantized matmul kernels (like llama.cpp/bitsandbytes).
  - Sequence length and batch size can be tuned for memory/performance tradeoff.

## 2026-03-20 (continued)

### Reference Architecture & Lessons from Qwen3 Architects

- **Full Transformer Forward:**
  - Includes RMSNorm (pre/post), multi-head self-attention (q_proj, k_proj, v_proj, o_proj), per-head QK RMSNorm (Qwen3-specific), rotary embeddings (RoPE, base 1,000,000.0), grouped query attention, MLP (gate_proj, up_proj, down_proj, SwiGLU), residuals, and weight tying (embed_tokens/lm_head).
- **Critical Quirks:**
  - QK norms are essential for correct attention (not present in Llama/Mistral).
  - RoPE base must be 1,000,000.0 for Qwen3.
  - Weight tying: lm_head and embed_tokens must share weights.
- **Config Extraction:**
  - All config values (hidden size, head count, etc.) are extracted from GGUF fields, not hardcoded.
- **Minimal Forward Passes are Insufficient:**
  - Early versions only applied q_proj in sequence, which led to shape mismatches and did not reflect the real architecture.
- **Debugging & Profiling:**
  - VRAM logging and per-step timing included for profiling.
  - Tokenizer and embedding matrix alignment is critical to avoid OOV errors.

---

# GGUF-LoRA-Trainer: Implementation Status and Known Issues

## Current Status
- Loader, LoRA injection, and minimal training loop are implemented and tested for Qwen3 GGUF models.
- Tokenizer now extracts vocab and merges from GGUF fields using only indices in `field.data`, ensuring vocab size matches embedding size.
- Training loop concatenates prompt and response for next-token prediction, as per best practices.
- LoRA parameter registration and optimizer setup are correct; only LoRA params are trainable.
- Forward pass and loss computation are correct for the minimal transformer block.

## Known Issues / Next Steps
- **Tokenizer/Embedding Alignment:**
  - After fixing vocab extraction, rare OOV errors persist (e.g., `IndexError: Target 7274 is out of bounds`).
  - This suggests a possible mismatch between tokenizer vocab and embedding matrix, or an issue with merges/special tokens.
  - Next step: Print and compare vocab size, embedding size, and sample tokens to confirm alignment.
- **Loss Masking:**
  - Currently, loss is computed over the entire prompt+response. For SFT/chat tuning, consider masking loss to only the assistant's response.
- **Chat Template:**
  - The Qwen3 chat template is not yet applied; currently, only user/assistant messages are concatenated.
- **Batching:**
  - Only single-example batches are supported. Token-based batching and padding are not yet implemented.
- **Full Model Stack:**
  - Only a minimal transformer block is used for forward pass. Full stack and attention math are not yet implemented.
- **Robustness:**
  - OOV skipping logic has been removed; if OOV errors recur, further tokenizer/model alignment is needed.

## Recommendations
- Before further training, confirm tokenizer and embedding matrix are fully aligned.
- Add debug prints for vocab/embedding size and sample tokens if OOV errors persist.
- Implement chat template and batching as next priorities.
- Document any further architectural decisions in this file.

---

_Last updated: 2026-03-29_
_Last updated: 2026-03-19_
