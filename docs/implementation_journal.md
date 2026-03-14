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

---

(Continue to log all major steps, decisions, and issues here as the project progresses.)
