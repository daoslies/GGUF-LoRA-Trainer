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


4. To train LoRA adapters:

```
source .venv/bin/activate && python3 train.py --model ../Models/Qwen3-4B-Instruct-2507-Q8_0.gguf --data sample_dataset.jsonl --rank 8 --output Qwen3-4B-lora_out.gguf
```


## Features
- Modular architecture registry (easy to add new models)
- Q8_0 quantization support (others: NotImplementedError)
- Loader builds nn.Module objects with correct frozen/trainable split
- LoRA injection and gradient flow tested end-to-end

## GGUF/LoRA Inspection & Debugging

To inspect GGUF metadata, tensor shapes, and LoRA adapter coverage, use the included inspector:

```
python3 gguf_inspector.py <path-to-model.gguf>
python3 gguf_inspector.py <lora.gguf> <reference_lora.gguf>
```
- Add `--json` for machine-readable output.
- Inspector highlights LoRA tensor shape/naming issues and GGUF metadata problems.

## LoRA GGUF Output Format
- LoRA tensors are written in GGUF as follows:
  - `.lora_a`: shape `[out_features, rank]` (from LoRA B, transposed if needed)
  - `.lora_b`: shape `[rank, in_features]` (from LoRA A, transposed if needed)
- This matches llama.cpp and reference GGUF LoRA adapters.
- Inspector and `quick_test.py` can be used to verify tensor shapes.

## Evaluation

A strict output format is used for LoRA evaluation:

```
word_or_phrase: #RRGGBB
```
- One word/phrase per line, colon, space, then hex color.
- No extra text or explanations.

Eval helpers:
```python
import re

def parse_output(text):
    pattern = re.compile(r'^(.+?):\s*(#[0-9A-Fa-f]{6})$', re.MULTILINE)
    matches = pattern.findall(text)
    return {word.strip(): colour for word, colour in matches}

def eval_format_compliance(output):
    parsed = parse_output(output)
    return len(parsed) > 0

def eval_format_strict(output):
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    parsed = parse_output(output)
    return len(parsed) == len(lines)
```

- `eval_format_strict` is the main metric: all output lines must match the format, no extra prose.

## Changelog
- Added robust GGUF LoRA tensor shape/naming logic (see `gguf_lora/writer.py`)
- Added `gguf_inspector.py` for metadata/tensor/coverage/warning inspection and comparison
- Added strict evaluation helpers for output format compliance
- Improved documentation and debugging workflow

## Requirements
- Python 3
- PyTorch
- GGUF Python library

See `docs/implementation_journal.md` for progress and design notes.
