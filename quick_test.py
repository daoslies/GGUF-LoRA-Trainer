import yaml
from gguf import GGUFReader

with open("config.yaml") as f:
    config = yaml.safe_load(f)
model_path = config["model_path"]
reader = GGUFReader(model_path)

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
    print(tensor.name, tensor.tensor_type, tensor.shape)