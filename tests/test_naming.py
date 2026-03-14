import unittest
import yaml
from gguf_lora.naming import get_name_map
from gguf_lora.architectures.qwen3 import QUANTIZED_TENSORS, NUM_LAYERS
from gguf import GGUFReader

with open("config.yaml") as f:
    config = yaml.safe_load(f)
MODEL_PATH = config["model_path"]

class TestNaming(unittest.TestCase):
    def test_registry_routes_qwen3_correctly(self):
        name_map = get_name_map(MODEL_PATH)
        self.assertEqual(name_map.ARCHITECTURE_ID, 'qwen3')

    def test_all_blocks_map_correctly(self):
        for i in range(NUM_LAYERS):
            for gguf_tmpl, hf_tmpl in QUANTIZED_TENSORS.items():
                gguf_name = gguf_tmpl.format(i=i)
                hf_name = hf_tmpl.format(i=i)
                self.assertIsNotNone(gguf_name)
                self.assertNotIn("layers.{i}", hf_name)

    def test_no_unmapped_tensors_in_real_file(self):
        reader = GGUFReader(MODEL_PATH)
        name_map = get_name_map(MODEL_PATH)
        unmapped = [t.name for t in reader.tensors if name_map.gguf_to_hf(t.name) is None]
        self.assertEqual(unmapped, [], f"Unmapped tensors: {unmapped}")

if __name__ == "__main__":
    unittest.main()
