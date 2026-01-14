#!/usr/bin/env python3
"""Train a BPE tokenizer using HuggingFace tokenizers (Rust-based, very fast).

This trains a 32k vocab tokenizer in minutes instead of hours.
"""

import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

PROJECT_ROOT = Path(__file__).parent.parent


def train_fast_tokenizer(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    max_files: int = None,
):
    """Train a BPE tokenizer using HuggingFace tokenizers library."""

    print(f"Training {vocab_size:,} vocab tokenizer with HuggingFace tokenizers (Rust)")
    print(f"Data directory: {data_dir}")

    # Get story files
    story_files = sorted(data_dir.glob("story_*.txt"))
    if max_files:
        story_files = story_files[:max_files]

    print(f"Using {len(story_files):,} story files")

    # Convert to string paths for tokenizers library
    file_paths = [str(f) for f in story_files]

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder for proper detokenization
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Trainer with special tokens
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    # Train!
    print(f"\nTraining tokenizer...")
    start_time = time.time()
    tokenizer.train(file_paths, trainer)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} minutes)")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved to: {tokenizer_path}")

    # Test encoding
    test_text = "Once upon a time, there was a little girl named Lily."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)

    print(f"\nTest encoding:")
    print(f"  Original: {test_text}")
    print(f"  Tokens: {len(encoded.ids)} tokens")
    print(f"  Token IDs: {encoded.ids[:20]}...")
    print(f"  Decoded: {decoded}")

    return tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train fast BPE tokenizer")
    parser.add_argument("--data-dir", type=str, default="data/tinystories_500k",
                        help="Directory containing story files")
    parser.add_argument("--output", type=str, default="models/tokenizer_32k",
                        help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files to use (default: all)")

    args = parser.parse_args()

    train_fast_tokenizer(
        data_dir=PROJECT_ROOT / args.data_dir,
        output_dir=PROJECT_ROOT / args.output,
        vocab_size=args.vocab_size,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
