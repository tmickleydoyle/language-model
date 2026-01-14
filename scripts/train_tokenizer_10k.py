#!/usr/bin/env python3
"""Train a 10k vocabulary BPE tokenizer on 25k TinyStories (50% of dataset)."""

import sys
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import BPETokenizer


def main():
    # Configuration
    DATA_DIR = PROJECT_ROOT / "data" / "tinystories_50k"
    OUTPUT_PATH = PROJECT_ROOT / "models" / "tokenizer_10k" / "tokenizer"
    VOCAB_SIZE = 10000
    MAX_STORIES = 25000  # Only use 50% to avoid over-learning patterns

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load story files (only first 25k)
    story_files = sorted(DATA_DIR.glob("story_*.txt"))[:MAX_STORIES]
    print(f"Using {len(story_files):,} story files (50% of dataset) from {DATA_DIR}")

    if len(story_files) == 0:
        raise ValueError(f"No story files found in {DATA_DIR}")

    # Load stories with progress
    print("Loading stories...")
    all_text = []
    for i, f in enumerate(story_files):
        with open(f, 'r', encoding='utf-8') as file:
            all_text.append(file.read().strip())
        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i+1:,}/{len(story_files):,} files...", flush=True)

    full_text = "\n\n".join(all_text)
    print(f"Total text: {len(full_text):,} characters")

    # Train tokenizer
    print(f"\nTraining {VOCAB_SIZE:,} vocab tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(full_text, max_vocab_size=VOCAB_SIZE, verbose=True, min_frequency=2)

    # Save tokenizer
    tokenizer.save(str(OUTPUT_PATH))
    print(f"\nSaved tokenizer to {OUTPUT_PATH}")
    print(f"Vocabulary size: {len(tokenizer.vocab):,} tokens")

    # Test encoding
    test_text = "Once upon a time, there was a little girl named Lily."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"\nTest encoding:")
    print(f"  Original: {test_text}")
    print(f"  Tokens: {len(tokens)} tokens")
    print(f"  Decoded: {decoded}")


if __name__ == "__main__":
    main()
