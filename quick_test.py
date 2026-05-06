import yaml
from gguf import GGUFReader

"""

check_lora = True

with open("config.yaml") as f:
    config = yaml.safe_load(f)
model_path = config["model_path"]
reader = GGUFReader(model_path)

# choose between model_path and lora_out_path via config
chosen_path = config.get("lora_out_path") if check_lora else model_path
if check_lora and not chosen_path:
    raise ValueError("config['use_lora_out_path'] is True but 'lora_out_path' is missing or empty")

# recreate reader only if we need to open a different file than the one used above
if chosen_path != model_path:
    reader = GGUFReader(chosen_path)

print(f"Using file: {chosen_path}")

print("GGUFReader attributes:", dir(reader))
for attr in dir(reader):
    if not attr.startswith("_"):
        try:
            value = getattr(reader, attr)
            print(f"reader.{attr}: type={type(value)} value={str(value)[:200]}")
            if hasattr(value, 'keys'):
                print(f"reader.{attr}.keys():", list(value.keys()))
        except Exception as e:
            print(f"reader.{attr}: Exception: {e}")

for tensor in reader.tensors:
    print(tensor.name, tensor.tensor_type, tensor.shape)"""



from gguf import GGUFReader
# enhanced debug prints for the readers/tensor checks

paths = [
    ("lora_out_2", "Qwen3-4B-lora_out_2.gguf"),
    ("lora_out_3", "Qwen3-4B-lora_out_3.gguf"),
    ("lora_out_4", "Qwen3-4B-lora_out_4.gguf"),
    ("merged_adapter", "/home/noli/Software/LLM/Models/Qwen3/qwen3-4b-webai-official-math-merged-adapter/qwen3-4B-webai-official-math-merged-adapter-F32-LoRA.gguf"),
]

for label, path in paths:
    print(f"\n--- Opening [{label}] -> {path}")
    try:
        reader = GGUFReader(path)
    except Exception as e:
        print(f"  ERROR: failed to open file: {e}")
        continue

    total_tensors = len(reader.tensors)
    print(f"  total tensors in file: {total_tensors}")

    matches = [t for t in reader.tensors if "blk.0.attn_k" in t.name]
    print(f"  tensors matching 'blk.0.attn_k': {len(matches)}")
    for t in matches:
        # print name, shape and tensor type for clarity
        print(f"    {t.name}: shape={t.shape} type={getattr(t, 'tensor_type', 'N/A')}")

    if not matches:
        print("    (no matching tensors found)")

# check base model for exact tensor name
base_path = "/home/noli/Software/LLM/Agentic/Models/Qwen3-4B-Instruct-2507-Q8_0.gguf"
print(f"\n--- Checking base model -> {base_path}")
try:
    reader = GGUFReader(base_path)
except Exception as e:
    print(f"  ERROR: failed to open base model file: {e}")
else:
    found = None
    for t in reader.tensors:
        if t.name == "blk.0.attn_k.weight":
            found = t
            break
    if found:
        print(f"  Found base model tensor: {found.name} shape={found.shape} type={getattr(found, 'tensor_type', 'N/A')}")
    else:
        print("  base model tensor 'blk.0.attn_k.weight' not found")