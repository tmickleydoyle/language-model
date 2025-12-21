#!/usr/bin/env python3
"""Interactive story generation with trained model."""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config
from src.model import create_model
from src.tokenizer import BPETokenizer


def load_model(model_dir: str):
    """Load model and tokenizer from directory."""
    model_path = Path(model_dir)

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(str(model_path / "tokenizer"))
    print(f"Loaded tokenizer: {tokenizer.vocab_size} tokens")

    # Load checkpoint
    checkpoint_path = model_path / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    # Get config from checkpoint
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        config = Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=saved_config.get("n_embd", 384),
            n_head=saved_config.get("n_head", 6),
            n_layer=saved_config.get("n_layer", 6),
            n_kv_head=saved_config.get("n_kv_head", 2),  # GQA KV heads
            block_size=saved_config.get("block_size", 256),
            dropout=saved_config.get("dropout", 0.2),
            attention_type=saved_config.get("attention_type", "gqa"),
            device=device,
        )
    else:
        # Default for small preset
        config = Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            n_kv_head=2,
            block_size=256,
            attention_type="gqa",
            device=device,
        )

    # Create and load model
    model = create_model(config, tokenizer.vocab_size)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(config.device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {params:,} parameters on {config.device}")

    return model, tokenizer, config


def generate(
    model,
    tokenizer,
    config,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    add_bos: bool = True,
) -> str:
    """Generate text from prompt.

    Args:
        model: The language model
        tokenizer: BPE tokenizer
        config: Model config
        prompt: Text prompt to continue
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        add_bos: Whether to prepend BOS token (start of story)
    """
    eos_token_id = tokenizer.eos_token_id  # 258

    # Encode prompt, optionally with BOS token
    if add_bos and hasattr(tokenizer, 'encode_with_special_tokens'):
        tokens = tokenizer.encode_with_special_tokens(prompt, add_bos=True, add_eos=False)
    else:
        tokens = tokenizer.encode(prompt)

    prompt_length = len(tokens)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate if needed
            if input_ids.shape[1] >= config.block_size:
                input_ids = input_ids[:, -config.block_size:]

            # Forward pass
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == eos_token_id:
                break

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode all tokens, skip special tokens
    all_tokens = input_ids[0].tolist()
    output = tokenizer.decode(all_tokens, skip_special_tokens=True)

    # The output should contain the prompt - return as-is
    # If decode lost the prompt somehow, prepend it
    if not output.strip():
        return prompt

    return output.strip()


def interactive_mode(model, tokenizer, config):
    """Run interactive generation."""
    print("\n" + "=" * 50)
    print("Interactive Story Generation")
    print("=" * 50)
    print("Commands:")
    print("  Type a prompt to generate a story")
    print("  'temp 0.5' - Change temperature (0.1-2.0)")
    print("  'tokens 50' - Change max tokens")
    print("  'quit' - Exit")
    print()
    print("The model will automatically:")
    print(f"  - Start with {BPETokenizer.BOS_TOKEN} (beginning of story)")
    print(f"  - Stop at {BPETokenizer.EOS_TOKEN} (end of story)")
    print("=" * 50)

    temperature = 0.8
    max_tokens = 100

    while True:
        try:
            user_input = input(f"\n[temp={temperature}, tokens={max_tokens}] > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.startswith("temp "):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"Temperature set to {temperature}")
                except (IndexError, ValueError):
                    print("Usage: temp 0.8")
                continue

            if user_input.startswith("tokens "):
                try:
                    max_tokens = int(user_input.split()[1])
                    max_tokens = max(10, min(500, max_tokens))
                    print(f"Max tokens set to {max_tokens}")
                except (IndexError, ValueError):
                    print("Usage: tokens 100")
                continue

            # Generate with BOS token
            print("\nGenerating...")
            output = generate(
                model, tokenizer, config,
                user_input,
                max_tokens,
                temperature,
                add_bos=True,  # Always add BOS for story generation
            )
            print(f"\n{output}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate stories with trained model")
    parser.add_argument("model", help="Path to model directory")
    parser.add_argument("-p", "--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("-t", "--temperature", type=float, default=0.8)
    parser.add_argument("-n", "--tokens", type=int, default=100, help="Max tokens")
    parser.add_argument("--no-bos", action="store_true", help="Don't add BOS token")

    args = parser.parse_args()

    # Load model
    model, tokenizer, config = load_model(args.model)

    if args.prompt:
        # Single generation
        output = generate(
            model, tokenizer, config,
            args.prompt,
            args.tokens,
            args.temperature,
            add_bos=not args.no_bos,
        )
        print(f"\n{output}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, config)


if __name__ == "__main__":
    main()
