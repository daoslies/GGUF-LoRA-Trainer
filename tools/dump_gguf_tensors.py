# Utility to dump all tensor names/types/shapes from a GGUF file
from gguf import GGUFReader
import sys

if len(sys.argv) != 2:
    print("Usage: python tools/dump_gguf_tensors.py <path-to-model.gguf>")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
for tensor in reader.tensors:
    print(tensor.name, tensor.tensor_type, tensor.shape)
