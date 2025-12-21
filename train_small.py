#!/usr/bin/env python3
"""Train a right-sized language model on focused domain data.

This script implements best practices for training small LLMs:
- Appropriate model size for data volume (~15-20M parameters)
- Domain-specific vocabulary (4-8K tokens)
- Perplexity and diversity tracking
- Sample generation during training
- Early stopping with patience
- MPS-optimized for Apple Silicon

Usage:
    # Train on a directory of text files
    python train_small.py data/python_docs -o python_model

    # Train on a single combined file
    python train_small.py data/combined.txt -o my_model --vocab-size 6000

    # Quick test with tiny model
    python train_small.py data/sample.txt -o test_model --preset tiny
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

# Force unbuffered output for real-time progress display when piped
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

    Note: BOS/EOS tokens are now added as token IDs during encoding,
    not as strings embedded in the text. This is handled by TextDataset.

    Args:
        data_path: Path to text file or directory of text files

    Returns:
        Combined text content with document separators
    """
    path = Path(data_path)

    # Use a simple separator that the model can learn
    DOC_SEP = "\n\n"

    if path.is_file():
        logger.info(f"Loading single file: {path}")
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read().strip()

    elif path.is_dir():
        logger.info(f"Loading directory: {path}")
        texts = []
        files = list(path.glob("**/*.txt"))
        # Skip combined file if present
        files = [f for f in files if not f.name.startswith("_")]
        logger.info(f"Found {len(files)} text files")

        for file_path in sorted(files):
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


def create_config_from_preset(
    preset: str,
    vocab_size: int,
    device: Optional[str] = None,
) -> Config:
    """Create configuration from preset name.

    Args:
        preset: Preset name ('small', 'tiny', or 'custom')
        vocab_size: Vocabulary size
        device: Device override

    Returns:
        Config object
    """
    if preset == "tiny":
        return Config.tiny_model_preset(vocab_size=vocab_size, device=device)
    elif preset == "small":
        return Config.small_model_preset(vocab_size=vocab_size, device=device)
    else:
        raise ValueError(f"Unknown preset: {preset}")


def estimate_optimal_vocab(text: str, target_tokens: int) -> int:
    """Estimate optimal vocabulary size based on data.

    Args:
        text: Training text
        target_tokens: Target token count

    Returns:
        Recommended vocabulary size
    """
    text_len = len(text)
    estimated_tokens = text_len // 4  # Rough estimate

    # Rule of thumb: vocab should be ~sqrt(tokens) to 2*sqrt(tokens)
    # But capped between 2000 and 10000 for small models
    import math
    optimal = int(math.sqrt(estimated_tokens) * 1.5)
    return max(2000, min(10000, optimal))


def split_data(text: str, val_ratio: float = 0.1) -> Tuple[str, str]:
    """Split text into train and validation sets.

    Args:
        text: Full text corpus
        val_ratio: Fraction for validation

    Returns:
        Tuple of (train_text, val_text)
    """
    split_idx = int(len(text) * (1 - val_ratio))
    return text[:split_idx], text[split_idx:]


def train_model(args) -> None:
    """Main training function."""
    print("=" * 60)
    print("Small Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    text = load_text_data(args.data)
    total_chars = len(text)
    estimated_tokens = total_chars // 4

    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: {estimated_tokens:,}")

    if total_chars < 10000:
        print("  WARNING: Very small dataset. Consider getting more data.")

    # 2. Determine vocabulary size and configure tokenizer
    print("\n[2/6] Configuring tokenizer...")
    tokenizer = BPETokenizer()

    if args.tokenizer_path:
        # Load existing tokenizer
        print(f"  Loading existing tokenizer from {args.tokenizer_path}...")
        tokenizer.load(args.tokenizer_path)
        actual_vocab = tokenizer.vocab_size
        print(f"  Loaded tokenizer with {actual_vocab} tokens")
    else:
        # Train new tokenizer
        if args.vocab_size:
            vocab_size = args.vocab_size
        else:
            vocab_size = estimate_optimal_vocab(text, estimated_tokens)
        print(f"  Target vocabulary size: {vocab_size}")

        if args.tokenizer_sample_ratio < 1.0:
            sample_size = int(len(text) * args.tokenizer_sample_ratio)
            tokenizer_text = text[:sample_size]
            print(f"  Training BPE tokenizer on {args.tokenizer_sample_ratio*100:.0f}% sample ({len(tokenizer_text):,} chars)...")
        else:
            tokenizer_text = text
            print("  Training BPE tokenizer on full dataset...")

        tokenizer.train(tokenizer_text, max_vocab_size=vocab_size, min_frequency=2)
        actual_vocab = tokenizer.vocab_size
        print(f"  Actual vocabulary size: {actual_vocab}")

    # Get coverage stats (skip if using existing tokenizer - encoding is slow)
    if args.tokenizer_path:
        # Estimate based on typical compression ratio (~2.5x for BPE)
        estimated_total_tokens = total_chars // 3
        print(f"  Estimated tokens: ~{estimated_total_tokens:,} (skipping full encode)")
    else:
        coverage = tokenizer.get_coverage_stats(text)
        print(f"  Compression ratio: {coverage['compression_ratio']:.2f}x")
        print(f"  Actual tokens: {coverage['total_tokens']:,}")
        estimated_total_tokens = coverage['total_tokens']

    # 3. Create configuration
    print("\n[3/6] Creating model configuration...")
    config = create_config_from_preset(args.preset, actual_vocab, args.device)

    # Estimate parameters
    param_estimate = (
        config.n_embd * actual_vocab +  # Embeddings
        config.n_layer * (
            4 * config.n_embd * config.n_embd +  # Attention
            8 * config.n_embd * config.n_embd    # FFN (approx for SwiGLU)
        )
    )
    tokens_per_param = estimated_total_tokens / param_estimate

    print(f"  Model preset: {args.preset}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Context size: {config.block_size}")
    print(f"  Estimated parameters: {param_estimate / 1e6:.1f}M")
    print(f"  Tokens per parameter: {tokens_per_param:.1f}")

    if tokens_per_param < 10:
        print("  WARNING: Low token/param ratio. Model may overfit.")
    elif tokens_per_param > 100:
        print("  NOTE: High token/param ratio. Could use larger model.")

    # 4. Create datasets
    print("\n[4/6] Creating datasets...", flush=True)
    train_text, val_text = split_data(text, val_ratio=0.1)

    print(f"  Creating train dataset ({len(train_text):,} chars)...", flush=True)
    train_dataset = TextDataset(
        text=train_text,
        tokenizer=tokenizer,
        block_size=config.block_size,
    )
    print(f"  Creating val dataset ({len(val_text):,} chars)...", flush=True)
    val_dataset = TextDataset(
        text=val_text,
        tokenizer=tokenizer,
        block_size=config.block_size,
    )

    print(f"  Train tokens: {len(train_dataset.tokens):,}", flush=True)
    print(f"  Val tokens: {len(val_dataset.tokens):,}", flush=True)

    # 5. Create model
    print("\n[5/6] Creating model...")
    model = create_model(config, actual_vocab)
    actual_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {actual_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {config.device}")

    # 6. Training loop
    print("\n[6/6] Training...")
    print("-" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=output_dir,
    )

    # Save tokenizer
    tokenizer.save(str(output_dir / "tokenizer"))
    print(f"  Tokenizer saved to {output_dir / 'tokenizer'}")

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    # Sample prompts for monitoring (default to story-style prompts)
    if args.prompts:
        sample_prompts = args.prompts.split(",")
    else:
        # TinyStories-style prompts
        sample_prompts = [
            "Once upon a time",
            "The little girl",
            "One day, a",
        ]

    for epoch in range(config.max_epochs):
        # Train one epoch
        train_loss = trainer.train_epoch()
        eval_results = trainer.evaluate()

        if isinstance(eval_results, dict):
            val_loss = eval_results.get("val_loss", float('inf'))
        else:
            val_loss = eval_results if eval_results else float('inf')

        # Compute perplexity
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
        if args.show_samples and (epoch + 1) % args.sample_interval == 0:
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

            # Compute diversity
            diversity = compute_diversity_metrics(samples)
            print(f"  Diversity: {diversity['repetition_score']:.2f}", flush=True)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_checkpoint(output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
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

    # Final samples
    if args.show_samples:
        print("\nFinal generation samples:")
        samples = trainer.sample_during_training(
            prompts=sample_prompts,
            num_tokens=50,
            temperature=0.7,
            log_samples=False,
        )
        for i, sample in enumerate(samples):
            print(f"\n  Prompt: '{sample_prompts[i]}'")
            print(f"  Output: {sample[:200]}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a small, focused language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on Python documentation
    python train_small.py data/python_docs -o python_model

    # Quick test with tiny model
    python train_small.py data/sample.txt -o test --preset tiny

    # Custom vocabulary size
    python train_small.py data/text.txt -o model --vocab-size 4000
        """
    )

    parser.add_argument(
        "data",
        help="Path to training data (file or directory of .txt files)"
    )
    parser.add_argument(
        "-o", "--output",
        default="small_model",
        help="Output directory for model and tokenizer"
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "small"],
        default="small",
        help="Model size preset: tiny (~5M params) or small (~15M params)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (auto-detected if not specified)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (auto-detected if not specified)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--tokenizer-sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of data for tokenizer training (default: 1.0 = full, use 0.1 for 10%% sample)"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to existing tokenizer to load (skips tokenizer training)"
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        default=True,
        help="Show generation samples during training"
    )
    parser.add_argument(
        "--no-samples",
        dest="show_samples",
        action="store_false",
        help="Disable generation samples during training"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Epochs between sample generation"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated prompts for sample generation"
    )

    args = parser.parse_args()

    try:
        train_model(args)
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
