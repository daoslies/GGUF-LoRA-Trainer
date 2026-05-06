import json
import re
import argparse
import requests
import yaml
from pathlib import Path
from collections import defaultdict

# ── Config loading ───────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)
ML_SERVER_PORT = config.get("ml_server", {}).get("port", 27776)
ML_SERVER_URL = f"http://localhost:{ML_SERVER_PORT}"

# ── Format parsing ───────────────────────────────────────────────────────────
ANNOTATION_PATTERN = re.compile(r'^(.+?):\s*(#[0-9A-Fa-f]{6})$', re.MULTILINE)

def parse_output(text):
    matches = ANNOTATION_PATTERN.findall(text)
    return {word.strip(): colour.upper() for word, colour in matches}

def eval_format_compliance(output):
    return len(parse_output(output)) > 0

def eval_strict_compliance(output):
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    parsed = parse_output(output)
    return len(lines) > 0 and len(parsed) == len(lines)

def eval_min_annotations(output, minimum=3):
    return len(parse_output(output)) >= minimum

# ── Colour consistency ───────────────────────────────────────────────────────
def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def colour_distance(hex1, hex2):
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    max_dist = (255**2 * 3) ** 0.5
    dist = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) ** 0.5
    return dist / max_dist

def eval_colour_consistency(all_outputs):
    word_colours = defaultdict(list)
    for output in all_outputs:
        parsed = parse_output(output)
        for word, colour in parsed.items():
            word_colours[word.lower()].append(colour)
    repeated = {w: colours for w, colours in word_colours.items() if len(colours) >= 2}
    if not repeated:
        return None, {}
    distances = []
    per_word = {}
    for word, colours in repeated.items():
        pairs = [(colours[i], colours[j])
                 for i in range(len(colours))
                 for j in range(i+1, len(colours))]
        avg_dist = sum(colour_distance(a, b) for a, b in pairs) / len(pairs)
        distances.append(avg_dist)
        per_word[word] = {
            "colours": colours,
            "avg_distance": round(avg_dist, 4),
            "consistent": avg_dist < 0.15
        }
    return round(sum(distances) / len(distances), 4) if distances else None, per_word

# ── Inference via ML Server ──────────────────────────────────────────────────
def load_model(model_path, lora_path=None):
    payload = {"model_path": model_path}
    if lora_path:
        payload["lora_path"] = lora_path
    resp = requests.post(f"{ML_SERVER_URL}/load_model", json=payload)
    if not resp.ok or not resp.json().get("success", True):
        raise RuntimeError(f"Failed to load model: {resp.text}")
    return resp.json()

def run_inference(prompt, max_tokens=300):
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    resp = requests.post(f"{ML_SERVER_URL}/inference", json=payload)
    if not resp.ok:
        raise RuntimeError(f"Inference failed: {resp.text}")
    return resp.json().get("output", resp.text)

INFERENCE_PROMPT_TEMPLATE = (
    "You are a colour annotator. Read the diary entry and identify emotionally or symbolically significant words and phrases. "
    "For each, assign a hex colour that captures how it feels in context. Output only lines in the format: word: #RRGGBB\n\n{diary_text}"
)

# ── Main eval loop ───────────────────────────────────────────────────────────
def run_eval(model_path, lora_path, val_data_path):
    with open(val_data_path) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    results = []
    all_outputs = []
    log_lines = []
    print(f"Evaluating {len(examples)} examples...")
    print(f"Model: {model_path}")
    print(f"LoRA:  {lora_path or 'none (baseline)'}\n")
    # Load model once
    try:
        load_model(model_path, lora_path)
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return {}
    for i, example in enumerate(examples):
        user_msg = next(m["content"] for m in example["messages"] if m["role"] == "user")
        reference_output = next(m["content"] for m in example["messages"] if m["role"] == "assistant")
        prompt = INFERENCE_PROMPT_TEMPLATE.format(diary_text=user_msg)
        log_lines.append(f"\n[{i+1}/{len(examples)}] Prompt:\n{user_msg}\n")
        try:
            output = run_inference(prompt)
        except Exception as e:
            print(f"ERROR: {e}")
            log_lines.append(f"  ERROR: {e}\n")
            continue
        compliant = eval_format_compliance(output)
        strict = eval_strict_compliance(output)
        min_ann = eval_min_annotations(output)
        n_ann = len(parse_output(output))
        all_outputs.append(output)
        result = {
            "input": user_msg,
            "output": output,
            "reference": reference_output,
            "format_compliant": compliant,
            "strict_compliant": strict,
            "min_annotations": min_ann,
            "n_annotations": n_ann,
        }
        results.append(result)
        status = "✓" if strict else ("~" if compliant else "✗")
        # Print more informative diagnostics if not strict
        if strict:
            print(f"{status} strict={strict} n={n_ann}")
            log_lines.append(f"  {status} strict={strict} n={n_ann}\n")
        else:
            print(f"{status} strict={strict} n={n_ann} compliant={compliant} min_ann={min_ann}")
            log_lines.append(f"  {status} strict={strict} n={n_ann} compliant={compliant} min_ann={min_ann}\n")
            if not compliant:
                print(f"    Output not format compliant. Raw output:\n    {repr(output)[:200]}")
                log_lines.append(f"    Output not format compliant. Raw output:\n    {repr(output)[:200]}\n")
            elif not min_ann:
                print(f"    Fewer than minimum annotations. Output:\n    {repr(output)[:200]}")
                log_lines.append(f"    Fewer than minimum annotations. Output:\n    {repr(output)[:200]}\n")
            elif not strict:
                lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
                parsed = parse_output(output)
                for idx, l in enumerate(lines):
                    if not ANNOTATION_PATTERN.match(l):
                        print(f"    Line {idx+1} not valid: {repr(l)}")
                        log_lines.append(f"    Line {idx+1} not valid: {repr(l)}\n")
    # Save log file for later analysis
    log_path = "eval_detailed_log.txt"
    with open(log_path, "w") as logf:
        logf.writelines(log_lines)
    print(f"\nDetailed per-example log written to {log_path}")
    n = len(results)
    if n == 0:
        print("No results.")
        return {}
    compliance_rate = sum(r["format_compliant"] for r in results) / n
    strict_rate = sum(r["strict_compliant"] for r in results) / n
    min_ann_rate = sum(r["min_annotations"] for r in results) / n
    avg_annotations = sum(r["n_annotations"] for r in results) / n
    avg_consistency, per_word_consistency = eval_colour_consistency(all_outputs)
    summary = {
        "model": model_path,
        "lora": lora_path,
        "n_examples": n,
        "scores": {
            "format_compliance":  round(compliance_rate, 3),
            "strict_compliance":  round(strict_rate, 3),
            "min_annotations":    round(min_ann_rate, 3),
            "avg_annotations":    round(avg_annotations, 1),
            "colour_consistency": avg_consistency,
        },
        "per_example": results,
        "colour_consistency_detail": per_word_consistency,
    }
    print(f"\n{'─'*50}")
    print(f"FORMAT COMPLIANCE:  {compliance_rate:.1%}  (any valid line produced)")
    print(f"STRICT COMPLIANCE:  {strict_rate:.1%}  ← the key metric")
    print(f"MIN ANNOTATIONS:    {min_ann_rate:.1%}  (≥3 words found)")
    print(f"AVG ANNOTATIONS:    {avg_annotations:.1f} words per entry")
    if avg_consistency is not None:
        print(f"COLOUR CONSISTENCY: {avg_consistency:.4f}  (lower = more consistent)")
    print(f"{'─'*50}")
    return summary

# ── Comparison ───────────────────────────────────────────────────────────────
def compare_results(baseline_path, lora_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(lora_path) as f:
        lora = json.load(f)
    print(f"\n{'═'*60}")
    print(f"  EVAL COMPARISON")
    print(f"{'═'*60}")
    print(f"{'Metric':<25} {'Baseline':>12} {'LoRA':>12} {'Delta':>10}")
    print(f"{'─'*60}")
    metrics = [
        ("strict_compliance", "Strict compliance", True),
        ("format_compliance", "Format compliance", True),
        ("min_annotations", "Min annotations", True),
        ("avg_annotations", "Avg annotations", True),
        ("colour_consistency", "Colour consistency", False),
    ]
    for key, label, higher_better in metrics:
        b = baseline["scores"].get(key)
        l = lora["scores"].get(key)
        if b is None or l is None:
            continue
        delta = l - b
        direction = "↑" if (delta > 0) == higher_better else "↓"
        sign = "+" if delta > 0 else ""
        if key in ("strict_compliance", "format_compliance", "min_annotations"):
            print(f"{label:<25} {b:>11.1%} {l:>11.1%} {direction} {sign}{delta:.1%}")
        else:
            print(f"{label:<25} {b:>12.4f} {l:>12.4f} {direction} {sign}{delta:.4f}")
    print(f"{'═'*60}")
    strict_b = baseline["scores"]["strict_compliance"]
    strict_l = lora["scores"]["strict_compliance"]
    delta = strict_l - strict_b
    if delta > 0.2:
        verdict = "✓ CLEAR IMPROVEMENT — LoRA is working"
    elif delta > 0.05:
        verdict = "~ MARGINAL IMPROVEMENT — more training may help"
    elif abs(delta) <= 0.05:
        verdict = "— NO MEANINGFUL CHANGE — check training loss"
    else:
        verdict = "✗ REGRESSION — something went wrong"
    print(f"\n  {verdict}\n")

# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model", nargs="?", help="Path to GGUF model (default: from config)")
    run_parser.add_argument("lora", nargs="?", help="Path to LoRA GGUF (default: from config)")
    run_parser.add_argument("--data", required=False, help="Path to eval dataset (default: from config)")
    run_parser.add_argument("--output", required=True)
    cmp_parser = subparsers.add_parser("compare")
    cmp_parser.add_argument("baseline")
    cmp_parser.add_argument("lora_results")
    args = parser.parse_args()
    if args.command == "run":
        model_path = args.model or config.get("model_path")
        lora_path = args.lora or config.get("lora_out_path")
        data_path = args.data or config.get("eval_dataset_path")
        if not model_path or not lora_path:
            print("ERROR: Must provide both model and lora paths (as args or in config.yaml)")
            exit(1)
        if not data_path:
            print("ERROR: No eval dataset path provided (set --data or eval_dataset_path in config.yaml)")
            exit(1)
        summary = run_eval(model_path, lora_path, data_path)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.output}")
    elif args.command == "compare":
        compare_results(args.baseline, args.lora_results)
    else:
        parser.print_help()
