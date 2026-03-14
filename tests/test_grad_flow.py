"""
Test gradient flow and parameter freezing for LoRA-injected GGUFLinear modules.
Optimized for speed: uses setUpClass to avoid repeated loader/injection setup.
"""
import torch
import unittest
import yaml
from gguf_lora.loader import GGUFLoader
from gguf_lora.lora import inject_lora, LoRAGGUFLinear

class TestLoRAGradFlowReal(unittest.TestCase):
    """
    Tests that gradients flow through LoRA-injected modules and only LoRA params are trainable.
    """
    @classmethod
    def setUpClass(cls):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        cls.loader = GGUFLoader(config["model_path"])
        cls.target_modules = cls.loader.name_map.DEFAULT_TARGET_MODULES
        inject_lora(cls.loader.modules, cls.target_modules, rank=4, alpha=8)

    def test_grad_flow_on_real_tensor(self):
        module = next((m for m in self.loader.modules.values() if isinstance(m, LoRAGGUFLinear)), None)
        if module is None:
            self.fail("No LoRAGGUFLinear found after injection")
        x = torch.randn(2, module.shape[1])
        out = module(x)
        out.sum().backward()
        self.assertIsNotNone(module.lora_A.grad, "lora_A grad missing")
        self.assertIsNotNone(module.lora_B.grad, "lora_B grad missing")
        trainable = [p for p in module.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2)

    def test_no_base_requires_grad(self):
        for name, mod in self.loader.modules.items():
            if isinstance(mod, LoRAGGUFLinear):
                for pname, param in mod.named_parameters():
                    if pname not in ("lora_A", "lora_B"):
                        self.assertFalse(param.requires_grad, f"{name}.{pname} should be frozen")

if __name__ == "__main__":
    unittest.main()
