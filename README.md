# GGUF-LoRA Trainer

Minimal library for training LoRA adapters directly on GGUF models (Qwen3-4B Q8_0 and similar), with modular architecture support and robust test coverage.

## Quick Start

1. Copy `config.yaml.example` to `config.yaml` and set your GGUF model path.
2. Run all tests:

```
python3 -m unittest discover tests
```

3. To inspect GGUF tensors:

```
python3 tools/dump_gguf_tensors.py <path-to-model.gguf>
```

## Features
- Modular architecture registry (easy to add new models)
- Q8_0 quantization support (others: NotImplementedError)
- Loader builds nn.Module objects with correct frozen/trainable split
- LoRA injection and gradient flow tested end-to-end

## Requirements
- Python 3
- PyTorch
- GGUF Python library

See `docs/implementation_journal.md` for progress and design notes.
