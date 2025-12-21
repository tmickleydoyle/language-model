#!/usr/bin/env python3
"""Continue training an existing model with more data.

This script loads a trained model checkpoint and continues training
on new data while keeping the same tokenizer and model architecture.

Usage:
    python continue_training.py prose_model_10k data/tinystories_50k -o prose_model_50k --epochs 30
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

from src.config import Config
from src.model import create_model
from src.data import TextDataset
from src.training.trainer import Trainer, compute_perplexity, compute_diversity_metrics
from src.tokenizer import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_text_data(data_path: str) -> str:
    """Load text data from file or directory.

    If directory contains _combined.txt, uses that for speed.
    Otherwise reads individual files.
    """
    path = Path(data_path)
    DOC_SEP = "\n\n"

    if path.is_file():
        logger.info(f"Loading single file: {path}")
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()

    elif path.is_dir():
        # Check for combined file first (much faster)
        combined_file = path / "_combined.txt"
        if combined_file.exists():
            logger.info(f"Loading combined file: {combined_file}")
            with open(combined_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read().strip()
            logger.info(f"Loaded {len(text):,} characters from combined file")
            return text

        # Fall back to individual files
        logger.info(f"Loading directory: {path}")
        texts = []
        files = list(path.glob("**/*.txt"))
        files = [f for f in files if not f.name.startswith("_")]
        logger.info(f"Found {len(files)} text files")

        for i, file_path in enumerate(sorted(files)):
            if (i + 1) % 5000 == 0:
                logger.info(f"  Read {i + 1}/{len(files)} files...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        return DOC_SEP.join(texts)
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")


def split_data(text: str, val_ratio: float = 0.1):
    """Split text into train and validation sets."""
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Continue training an existing model with more data"
    )
    parser.add_argument("model_dir", help="Path to existing model directory")
    parser.add_argument("data", help="Path to new training data")
    parser.add_argument("-o", "--output", required=True, help="Output directory for continued model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--sample-interval", type=int, default=5, help="Epochs between samples")

    args = parser.parse_args()

    print("=" * 60)
    print("Continue Training from Checkpoint")
    print("=" * 60)

    # 1. Load existing tokenizer
    print("\n[1/5] Loading existing tokenizer...")
    model_path = Path(args.model_dir)
    tokenizer = BPETokenizer()
    tokenizer.load(str(model_path / "tokenizer"))
    print(f"  Loaded tokenizer with {tokenizer.vocab_size} tokens")

    # 2. Load existing model checkpoint
    print("\n[2/5] Loading existing model...")
    checkpoint_path = model_path / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    saved_config = checkpoint["config"]

    config = Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=saved_config.get("n_embd", 256),
        n_head=saved_config.get("n_head", 4),
        n_layer=saved_config.get("n_layer", 4),
        n_kv_head=saved_config.get("n_kv_head", 1),
        block_size=saved_config.get("block_size", 128),
        dropout=saved_config.get("dropout", 0.1),
        attention_type=saved_config.get("attention_type", "gqa"),
        device=device,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )

    model = create_model(config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model: {params:,} parameters")
    print(f"  Architecture: {config.n_layer}L-{config.n_embd}D-{config.n_head}H")
    print(f"  Device: {device}")

    # 3. Load new training data
    print("\n[3/5] Loading new training data...", flush=True)
    text = load_text_data(args.data)
    print(f"  Total characters: {len(text):,}", flush=True)
    print(f"  Estimated tokens: ~{len(text) // 4:,}", flush=True)

    # 4. Create datasets
    print("\n[4/5] Creating datasets...", flush=True)
    train_text, val_text = split_data(text, val_ratio=0.1)

    print(f"  Encoding train data ({len(train_text):,} chars)...", flush=True)
    train_dataset = TextDataset(
        text=train_text,
        tokenizer=tokenizer,
        block_size=config.block_size,
        add_special_tokens=True,
    )
    print(f"  Train tokens: {len(train_dataset.tokens):,}", flush=True)

    print(f"  Encoding val data ({len(val_text):,} chars)...", flush=True)
    val_dataset = TextDataset(
        text=val_text,
        tokenizer=tokenizer,
        block_size=config.block_size,
        add_special_tokens=True,
    )
    print(f"  Val tokens: {len(val_dataset.tokens):,}", flush=True)
    print(f"  Token/param ratio: {len(train_dataset.tokens) / params:.1f}", flush=True)

    # 5. Continue training
    print("\n[5/5] Continuing training...")
    print("-" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer to new output
    tokenizer.save(str(output_dir / "tokenizer"))

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=output_dir,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    sample_prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a boy",
    ]

    for epoch in range(config.max_epochs):
        train_loss = trainer.train_epoch()
        eval_results = trainer.evaluate()

        val_loss = eval_results.get("val_loss", float('inf')) if isinstance(eval_results, dict) else eval_results

        train_ppl = compute_perplexity(train_loss)
        val_ppl = compute_perplexity(val_loss)

        elapsed = time.time() - start_time
        lr = trainer.optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:3d}/{config.max_epochs} | "
            f"Train: {train_loss:.4f} (ppl: {train_ppl:.1f}) | "
            f"Val: {val_loss:.4f} (ppl: {val_ppl:.1f}) | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.0f}s",
            flush=True
        )

        # Generate samples periodically
        if (epoch + 1) % args.sample_interval == 0:
            print("  Samples:", flush=True)
            samples = trainer.sample_during_training(
                prompts=sample_prompts,
                num_tokens=30,
                temperature=0.8,
                log_samples=False,
            )
            for i, sample in enumerate(samples):
                display = sample[:80].replace('\n', ' ')
                print(f"    [{i+1}] {display}...", flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_checkpoint(output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}", flush=True)
                break

    # Final results
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation perplexity: {compute_perplexity(best_val_loss):.1f}")
    print(f"  Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
