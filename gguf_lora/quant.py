import torch
from gguf import GGMLQuantizationType

def dequantize_q8_0(data: bytes, shape: tuple) -> torch.Tensor:
    """
    Dequantizes Q8_0 quantized bytes into a float32 tensor.
    Each block: [fp16 scale][32 x int8 values] = 34 bytes per block.
    """
    block_size = 34
    num_blocks = (len(data) // block_size)
    expected_elements = shape[0] * shape[1]
    if len(data) % block_size != 0:
        raise ValueError("Data size is not a multiple of Q8_0 block size (34 bytes)")
    # Use bytearray(data) to avoid buffer warnings (mmap safety)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    buf = buf.view(num_blocks, block_size)
    # Correctly extract fp16 scales from first 2 bytes of each block
    scales_bytes = buf[:, :2].contiguous().view(-1)
    scales = scales_bytes.view(torch.int16).view(torch.float16).to(torch.float32)
    values = buf[:, 2:].to(torch.int8).to(torch.float32)
    out = (values * scales.unsqueeze(1)).reshape(-1)
    out = out[:shape[0]*shape[1]].reshape(shape)
    return out

def dequantize(data: bytes, quant_type: str, shape: tuple) -> torch.Tensor:
    if normalise_quant_type(quant_type) == "Q8_0":
        return dequantize_q8_0(data, shape)
    raise NotImplementedError("Only Q8_0 quantization is supported in v0.1. Please use a Q8_0 model.")

def is_supported_quant_type(quant_type) -> bool:
    return quant_type in (GGMLQuantizationType.Q8_0, 8, "Q8_0")

def normalise_quant_type(quant_type) -> str:
    if quant_type in (GGMLQuantizationType.Q8_0, 8, "Q8_0"):
        return "Q8_0"
    raise NotImplementedError(f"Unsupported quant type: {quant_type}")
