import unittest
import yaml
import torch
import torch.nn as nn
from gguf_lora.loader import GGUFLoader
from gguf_lora.lora import LazyGGUFLinear
from gguf import GGUFReader

with open("config.yaml") as f:
    config = yaml.safe_load(f)
MODEL_PATH = config["model_path"]

class TestGGUFLoader(unittest.TestCase):
    def setUp(self):
        self.loader = GGUFLoader(MODEL_PATH)
        self.reader = self.loader.reader
        self.name_map = self.loader.name_map

    def test_tensor_mapping_consistency(self):
        # All mapped HF names should be unique
        hf_names = self.loader.list_hf_tensors()
        self.assertEqual(len(hf_names), len(set(hf_names)))

    def test_all_mapped_tensors_exist_in_reader(self):
        # All mapped GGUF names should exist in the GGUFReader tensors
        gguf_names = set(t.name for t in self.reader.tensors)
        for meta in self.loader.tensors.values():
            self.assertIn(meta['gguf_name'], gguf_names)

    def test_no_unmapped_lora_targets(self):
        # All LoRA target tensors in the GGUF should be mapped
        lora_targets = set()
        for i in range(self.name_map.NUM_LAYERS):
            for gguf_pat in self.name_map.QUANTIZED_TENSORS.keys():
                lora_targets.add(gguf_pat.format(i=i))
        present = set(t.name for t in self.reader.tensors)
        missing = [n for n in lora_targets if n in present and self.name_map.gguf_to_hf(n) is None]
        self.assertEqual(missing, [], f"Unmapped LoRA targets: {missing}")

    def test_f32_tensors_are_plain_parameters(self):
        # norms and q/k_norms should be nn.Parameter, not LazyGGUFLinear
        norm = self.loader.get_module("model.layers.0.input_layernorm.weight")
        self.assertIsInstance(norm, nn.Parameter)
        self.assertFalse(norm.requires_grad)

    def test_quantized_tensors_are_lazy_linear(self):
        q_proj = self.loader.get_module("model.layers.0.self_attn.q_proj.weight")
        self.assertIsInstance(q_proj, LazyGGUFLinear)

    def test_no_data_loaded_as_grad_enabled(self):
        # Nothing in the base model should have requires_grad=True
        # That's exclusively for LoRA matrices added later
        for name, module in self.loader.modules.items():
            for param in module.parameters() if hasattr(module, 'parameters') else []:
                self.assertFalse(param.requires_grad, f"{name} has requires_grad=True")

if __name__ == "__main__":
    unittest.main()
