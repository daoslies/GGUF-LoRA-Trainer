import re
import yaml
from gguf import GGUFReader
from typing import Dict, Any, List, Optional

def try_decode(val):
    # Try to decode lists of ints as utf-8 strings, else return as-is
    if isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
        try:
            return bytes(val).decode('utf-8')
        except Exception:
            return str(val)
    return val

def parse_fields(reader) -> Dict[str, Any]:
    fields = {}
    for name, field in reader.fields.items():
        try:
            val = field.parts[field.data[0]] if field.data else None
            fields[name] = try_decode(val)
        except Exception:
            fields[name] = "<unreadable>"
    return fields

def summarize_tensor_patterns(reader) -> Dict[str, int]:
    patterns = {}
    for t in reader.tensors:
        pattern = re.sub(r'\d+', 'N', t.name)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    return patterns

def infer_lora_ranks(reader) -> Dict[str, int]:
    ranks = {}
    for t in reader.tensors:
        if 'lora_a' in t.name:
            ranks[t.name] = min(t.shape)
    return ranks

def lora_coverage(reader) -> List[str]:
    layers = []
    for t in reader.tensors:
        if 'lora_a' in t.name:
            layers.append(t.name)
    return layers

def check_warnings(fields: Dict[str, Any], ranks: Dict[str, int], patterns: Dict[str, int]) -> List[str]:
    warnings = []
    if 'adapter.type' not in fields or fields['adapter.type'] != 'lora':
        warnings.append("'adapter.type' field missing or not 'lora' — may not load in llama.cpp")
    if 'general.architecture' not in fields:
        warnings.append("'general.architecture' field missing")
    if len(set(ranks.values())) > 1:
        warnings.append("Inconsistent LoRA rank across layers")
    if not any('lora_a' in k for k in patterns):
        warnings.append("No LoRA tensors found (no 'lora_a' patterns)")
    return warnings

def inspect_gguf(path: str) -> Dict[str, Any]:
    reader = GGUFReader(path)
    fields = parse_fields(reader)
    patterns = summarize_tensor_patterns(reader)
    ranks = infer_lora_ranks(reader)
    lora_layers = lora_coverage(reader)
    warnings = check_warnings(fields, ranks, patterns)
    return {
        "file": path,
        "metadata": fields,
        "tensor_summary": {
            "total": len(reader.tensors),
            "patterns": patterns
        },
        "lora_info": {
            "rank": list(set(ranks.values()))[0] if ranks else None,
            "covered_layers": lora_layers,
            "missing_layers": []  # For future: compare to reference
        },
        "warnings": warnings
    }

def print_inspection(result: Dict[str, Any]):
    print(f"\nFile: {result['file']}")
    print("\nMetadata (LoRA-relevant fields):")
    for k in sorted(result['metadata']):
        if any(x in k for x in ['lora', 'adapter', 'training', 'general']):
            v = result['metadata'][k]
            print(f"  {k:30s}: {v}")
    print("\nTensor patterns:")
    for p, count in sorted(result['tensor_summary']['patterns'].items()):
        print(f"  {count:3d}x  {p}")
    print("\nInferred LoRA ranks:")
    for layer in result['lora_info']['covered_layers']:
        print(f"  {layer}: rank={result['lora_info']['rank']}")
    if result['warnings']:
        print("\nWarnings:")
        for w in result['warnings']:
            print(f"  ⚠ {w}")

def compare_inspections(res1: Dict[str, Any], res2: Dict[str, Any]):
    print(f"\nComparing: {res1['file']}  <->  {res2['file']}")
    # Metadata diff (LoRA-relevant fields)
    keys1 = {k for k in res1['metadata'] if any(x in k for x in ['lora', 'adapter', 'training', 'general'])}
    keys2 = {k for k in res2['metadata'] if any(x in k for x in ['lora', 'adapter', 'training', 'general'])}
    all_keys = sorted(keys1 | keys2)
    print("\nMetadata differences:")
    for k in all_keys:
        v1 = res1['metadata'].get(k, '<missing>')
        v2 = res2['metadata'].get(k, '<missing>')
        if v1 != v2:
            print(f"  {k:30s}: {v1}   <->   {v2}")
    # Tensor pattern diff
    pats1 = res1['tensor_summary']['patterns']
    pats2 = res2['tensor_summary']['patterns']
    all_pats = sorted(set(pats1) | set(pats2))
    print("\nTensor pattern differences:")
    for p in all_pats:
        c1 = pats1.get(p, 0)
        c2 = pats2.get(p, 0)
        if c1 != c2:
            print(f"  {p:40s}: {c1:3d}   <->   {c2:3d}")
    # LoRA rank diff
    r1 = res1['lora_info']['rank']
    r2 = res2['lora_info']['rank']
    print("\nLoRA rank:")
    print(f"  {res1['file']}: {r1}")
    print(f"  {res2['file']}: {r2}")
    if r1 != r2:
        print("  ⚠ LoRA rank mismatch!")
    # LoRA coverage diff
    set1 = set(res1['lora_info']['covered_layers'])
    set2 = set(res2['lora_info']['covered_layers'])
    only1 = sorted(set1 - set2)
    only2 = sorted(set2 - set1)
    print("\nLoRA layer coverage differences:")
    if only1:
        print(f"  Layers only in {res1['file']}:")
        for l in only1:
            print(f"    {l}")
    if only2:
        print(f"  Layers only in {res2['file']}:")
        for l in only2:
            print(f"    {l}")
    if not only1 and not only2:
        print("  (identical coverage)")
    # Warnings
    print("\nWarnings:")
    for w in res1['warnings']:
        print(f"  {res1['file']}: ⚠ {w}")
    for w in res2['warnings']:
        print(f"  {res2['file']}: ⚠ {w}")

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python gguf_inspector.py <gguf_path> [<gguf_path2>] [--json]")
        sys.exit(1)
    use_json = '--json' in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(paths) == 0:
        print("Usage: python gguf_inspector.py <gguf_path> [<gguf_path2>] [--json]")
        sys.exit(1)
    result1 = inspect_gguf(paths[0])
    if len(paths) == 1:
        if use_json:
            print(json.dumps(result1, indent=2, ensure_ascii=False))
        else:
            print_inspection(result1)
    else:
        result2 = inspect_gguf(paths[1])
        if use_json:
            print(json.dumps({'file1': result1, 'file2': result2}, indent=2, ensure_ascii=False))
        else:
            compare_inspections(result1, result2)
