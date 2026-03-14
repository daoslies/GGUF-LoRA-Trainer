import unittest
import torch
from gguf_lora.quant import dequantize_q8_0

class TestQuant(unittest.TestCase):
    def test_dequant_output_is_plain_tensor(self):
        # Create fake Q8_0 data: 1 block, scale=1.0, values=range(-16,16)
        scale = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)  # 2 bytes
        values = torch.arange(-16, 16, dtype=torch.int8).to(torch.uint8)    # 32 bytes
        block = torch.cat([scale, values])
        data = bytes(block.tolist())
        out = dequantize_q8_0(data, (1, 32))
        self.assertIs(type(out), torch.Tensor)
        self.assertIn(out.dtype, [torch.float32, torch.float16])

    def test_q8_0_roundtrip_error(self):
        # Simulate quantize -> dequantize roundtrip
        # For this test, quantize is just scale=1.0, values=original rounded
        original = torch.randn(2, 32) * 10
        scales = torch.tensor([1.0, 1.0], dtype=torch.float16).view(-1, 1).repeat(1, 2).view(-1).to(torch.uint8)
        # For each row, get 2 bytes for scale, then 32 bytes for values
        scale0 = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        scale1 = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        values0 = original[0].round().clamp(-128,127).to(torch.int8).to(torch.uint8)
        values1 = original[1].round().clamp(-128,127).to(torch.int8).to(torch.uint8)
        block0 = torch.cat([scale0, values0])
        block1 = torch.cat([scale1, values1])
        blocks = torch.cat([block0, block1])
        data = bytes(blocks.tolist())
        recovered = dequantize_q8_0(data, (2, 32))
        error = (original.round() - recovered).abs().mean().item()
        self.assertLess(error, 0.01)

if __name__ == "__main__":
    unittest.main()
