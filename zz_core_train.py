import argparse
import yaml
import torch
import json
import time
import platform
from gguf_lora.loader import GGUFLoader
from gguf_lora.lora import inject_lora, LoRAGGUFLinear
from gguf_lora.writer import save_lora_gguf
from gguf_lora.utils import GGUFTokenizer
from gguf_lora.model_loaders import Qwen3ForCausalLM
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters on a GGUF model.")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--data", required=True, help="Path to training data (jsonl or similar)")
    parser.add_argument("--rank", type=int, required=True, help="LoRA rank")
    parser.add_argument("--output", required=True, help="Output GGUF LoRA file")
    return parser.parse_args()

def print_vram(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{prefix} VRAM: allocated={allocated:.2f} MiB, reserved={reserved:.2f} MiB")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}, platform: {platform.platform()}")
    loader = GGUFLoader(args.model)
    target_modules = loader.name_map.DEFAULT_TARGET_MODULES
    inject_lora(loader.modules, target_modules, rank=args.rank, alpha=args.rank*2)

    # Build Qwen3 model (full forward, LoRA-aware)
    config = loader.get_qwen3_config()
    model = Qwen3ForCausalLM(loader.modules, config).to(device)
    # Ensure all loader.modules tensors/parameters are on the same device
    for k, v in loader.modules.items():
        if isinstance(v, torch.nn.Parameter) or isinstance(v, torch.Tensor):
            loader.modules[k] = v.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)}")

    # Load tokenizer from GGUF
    tokenizer = GGUFTokenizer(args.model)
    # Load real dataset (expects jsonl with 'messages' and 'assistant' content)
    with open(args.data) as f:
        dataset = [json.loads(l) for l in f if l.strip()]
    # For demonstration: use all examples, or sample/batch as needed
    # We'll use the first example for shape check and a single step
    example = dataset[0]
    user_msg = next(m["content"] for m in example["messages"] if m["role"] == "user")
    target = next(m["content"] for m in example["messages"] if m["role"] == "assistant")

    # Tokenize input and target
    x_ids = tokenizer.encode_tensor(user_msg)
    y_ids = tokenizer.encode_tensor(target)
    # Uncomment for debugging tokenization issues:
    # print("Sample training input (tokens):", x_ids.tolist())
    # print("Sample training target (tokens):", y_ids.tolist())
    # print(f"[DEBUG] x_ids min: {x_ids.min().item()}, max: {x_ids.max().item()}")
    # print(f"[DEBUG] y_ids min: {y_ids.min().item()}, max: {y_ids.max().item()}")

    # Training loop skeleton
    epochs = 1  # Set to >1 for real training
    batch_size = 1  # For now, process one example at a time (token batching later)
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    scaler = GradScaler() if torch.cuda.is_available() else None
    print_vram("After model init")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for i, example in enumerate(dataset):
            step_start = time.time()
            # --- Apply Qwen3 chat template here (stub: just use user_msg/target for now) ---
            user_msg = next(m["content"] for m in example["messages"] if m["role"] == "user")
            target = next(m["content"] for m in example["messages"] if m["role"] == "assistant")
            x_ids = tokenizer.encode_tensor(user_msg)
            y_ids = tokenizer.encode_tensor(target)
            # --- Concatenate prompt and response for next-token prediction ---
            full_ids = torch.cat([x_ids, y_ids], dim=0)
            if full_ids.shape[0] < 2:
                continue  # skip too-short examples
            input_ids = full_ids[:-1].to(device)
            target_ids = full_ids[1:].to(device)
            # --- DEBUG: Print vocab/embedding alignment info ---
            if i == 0 and epoch == 0:
                print(f"[DEBUG] Tokenizer vocab size: {len(tokenizer.vocab)}")
                print(f"[DEBUG] Embedding matrix size: {model.embedding.shape[0]}")
                print(f"[DEBUG] Max target_id in batch: {target_ids.max().item()}")
                print(f"[DEBUG] First 10 vocab tokens: {tokenizer.vocab[:10]}")
            # --- Forward pass with mixed precision ---
            if torch.cuda.is_available():
                with autocast():
                    logits = model(input_ids)
                    logits = logits.view(-1, logits.shape[-1])
                    target_ids = target_ids.view(-1)
                    loss = loss_fn(logits, target_ids)
            else:
                logits = model(input_ids)
                logits = logits.view(-1, logits.shape[-1])
                target_ids = target_ids.view(-1)
                loss = loss_fn(logits, target_ids)
            print_vram("After forward")
            try:
                if target_ids.max().item() >= logits.shape[-1]:
                    print(f"[ERROR] OOV in batch {i}: max target_id {target_ids.max().item()} >= vocab size {logits.shape[-1]}")
                    print(f"[ERROR] Offending target_ids: {target_ids[target_ids >= logits.shape[-1]].tolist()}")
                    print(f"[ERROR] All target_ids: {target_ids.tolist()}")
            except Exception as e:
                print(f"[EXCEPTION] in batch {i}: {e}")
                print(f"[EXCEPTION] target_ids: {target_ids.tolist()}")
                print(f"[EXCEPTION] logits.shape: {logits.shape}")
                raise
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                print_vram("After backward")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                print_vram("After backward")
                optimizer.step()
            print(f"Step {i}: loss={loss.item():.4f} | step_time={time.time() - step_start:.2f}s | device={device}")

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
