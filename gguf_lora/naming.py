from gguf_lora.architectures.qwen3 import Qwen3NameMap

# Map GGUF architecture codes (int or str) to registry keys
ARCHITECTURE_CODE_MAP = {
    4: "qwen3",   # Qwen3-4B Q8_0 GGUF uses 4 for architecture
    "4": "qwen3", # Defensive: handle stringified int
    "qwen3": "qwen3",
}

ARCHITECTURE_REGISTRY = {
    "qwen3": Qwen3NameMap,
    # "qwen35": Qwen35NameMap,   # v0.2
    # "llama": LlamaNameMap,     # v0.2
}

def read_gguf_architecture_from_reader(reader) -> str:
    fields = getattr(reader, 'fields', None)
    if fields and 'general.architecture' in fields:
        field = fields['general.architecture']
        if hasattr(field, 'data') and isinstance(field.data, (list, tuple)) and len(field.data) > 0:
            arch_val = field.data[0]
            return ARCHITECTURE_CODE_MAP.get(arch_val, str(arch_val).lower())
        raise RuntimeError(
            f"Found 'general.architecture' but could not parse its value. "
            f"Raw field: {field}. Please open an issue with your GGUF file details."
        )
    raise RuntimeError("Could not find 'general.architecture' in GGUF fields.")

def get_name_map_from_reader(reader):
    arch = read_gguf_architecture_from_reader(reader)
    if arch not in ARCHITECTURE_REGISTRY:
        raise NotImplementedError(
            f"Architecture not supported. "
            f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}. "
            f"See CONTRIBUTING.md to add support."
        )
    return ARCHITECTURE_REGISTRY[arch]()

def read_gguf_architecture(gguf_path: str) -> str:
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    return read_gguf_architecture_from_reader(reader)

def get_name_map(gguf_path: str):
    arch = read_gguf_architecture(gguf_path)
    if arch not in ARCHITECTURE_REGISTRY:
        raise NotImplementedError(
            f"Architecture not supported. "
            f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}. "
            f"See CONTRIBUTING.md to add support."
        )
    return ARCHITECTURE_REGISTRY[arch]()
