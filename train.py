import argparse
import yaml
import torch
from gguf_lora.loader import GGUFLoader
from gguf_lora.lora import inject_lora, LoRAGGUFLinear
from gguf_lora.writer import save_lora_gguf

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters on a GGUF model.")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--data", required=True, help="Path to training data (jsonl or similar)")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank")
    parser.add_argument("--output", required=True, help="Output GGUF LoRA file")
    return parser.parse_args()

def main():
    args = parse_args()
    loader = GGUFLoader(args.model)
    target_modules = loader.name_map.DEFAULT_TARGET_MODULES
    inject_lora(loader.modules, target_modules, rank=args.rank, alpha=args.rank*2)

    # Assert only LoRA params are trainable
    trainable = [p for m in loader.modules.values() if isinstance(m, LoRAGGUFLinear) for p in m.parameters() if p.requires_grad]
    assert all(pname in ("lora_A", "lora_B") for m in loader.modules.values() if isinstance(m, LoRAGGUFLinear) for pname, p in m.named_parameters() if p.requires_grad), "Non-LoRA params are trainable!"
    print(f"Trainable params: {sum(p.numel() for p in trainable)}")

    # Dummy data: single batch, correct input shape for one LoRA-injected module
    lora_mod = next((m for m in loader.modules.values() if isinstance(m, LoRAGGUFLinear)), None)
    if lora_mod is None:
        raise RuntimeError("No LoRAGGUFLinear found after injection")
    x = torch.randn(2, lora_mod.shape[1])
    y = torch.randn(2, lora_mod.shape[0])

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    lora_mod.train()
    for step in range(1):
        optimizer.zero_grad()
        out = lora_mod(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step}: loss={loss.item():.4f}")

    # Save LoRA adapters to GGUF using writer.py
    save_lora_gguf(
        loader.modules,
        loader.name_map,
        args.output,
        alpha=args.rank*2,  # match alpha used in inject_lora
        arch_id=loader.name_map.ARCHITECTURE_ID
    )
    print(f"Saved LoRA GGUF to {args.output}")

if __name__ == "__main__":
    main()
