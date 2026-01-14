#!/usr/bin/env python3
"""Generate responses from instruction-following model."""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import Config
from src.model import create_model
from src.tokenizer_hf import HFTokenizer


def load_model(model_dir: str):
    """Load model and tokenizer from directory."""
    model_path = Path(model_dir)

    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No tokenizer found at {tokenizer_path}")

    tokenizer = HFTokenizer(str(tokenizer_path))
    print(f"Loaded tokenizer: {tokenizer.vocab_size} tokens")

    # Load checkpoint
    checkpoint_path = model_path / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: {checkpoint_path.name}")

    # Get config
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if "config" in checkpoint:
        saved_config = checkpoint["config"]
        config = Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=saved_config.get("n_embd", 384),
            n_head=saved_config.get("n_head", 8),
            n_layer=saved_config.get("n_layer", 6),
            n_kv_head=saved_config.get("n_kv_head", 2),
            block_size=saved_config.get("block_size", 1024),
            dropout=saved_config.get("dropout", 0.2),
            attention_type=saved_config.get("attention_type", "gqa"),
            device=device,
        )
    else:
        config = Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=384,
            n_head=8,
            n_layer=6,
            n_kv_head=2,
            block_size=1024,
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


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format instruction and input into prompt."""
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    return prompt


def generate(
    model,
    tokenizer,
    config,
    instruction: str,
    input_text: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """Generate response to instruction."""
    # Format prompt
    prompt = format_prompt(instruction, input_text)

    # Tokenize with BOS
    tokens = [tokenizer.BOS_TOKEN_ID] + tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

    eos_token_id = tokenizer.EOS_TOKEN_ID

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate if needed
            if input_ids.shape[1] >= config.block_size:
                input_ids = input_ids[:, -config.block_size:]

            # Forward
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check EOS
            if next_token.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode response only (after ### Response:)
    all_tokens = input_ids[0].tolist()
    full_text = tokenizer.decode(all_tokens, skip_special_tokens=True)

    # Extract response
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text

    return response


def main():
    parser = argparse.ArgumentParser(description="Generate responses from instruction model")
    parser.add_argument("--model", default="instruct_model", help="Path to model directory")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction to follow")
    parser.add_argument("--input", default="", help="Optional input data")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-n", "--tokens", type=int, default=256, help="Max tokens")

    args = parser.parse_args()

    # Load model
    model, tokenizer, config = load_model(args.model)

    # Generate
    print("\n" + "=" * 50)
    print(f"Instruction: {args.instruction}")
    if args.input:
        print(f"Input: {args.input}")
    print("=" * 50)

    response = generate(
        model, tokenizer, config,
        args.instruction,
        args.input,
        args.tokens,
        args.temperature,
    )

    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
