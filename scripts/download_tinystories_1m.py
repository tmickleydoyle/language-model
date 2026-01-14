#!/usr/bin/env python3
"""Download 1M TinyStories from HuggingFace."""

import os
import sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "tinystories_1m"


def main():
    print("Downloading TinyStories dataset from HuggingFace...")

    # Load the dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Full dataset: {len(dataset):,} stories")

    # Take first 1M stories
    num_stories = 1_000_000
    dataset = dataset.select(range(min(num_stories, len(dataset))))
    print(f"Selected: {len(dataset):,} stories")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save each story as a separate file
    print(f"\nSaving to {OUTPUT_DIR}...")
    for i, example in enumerate(dataset):
        story_file = OUTPUT_DIR / f"story_{i:07d}.txt"
        with open(story_file, 'w', encoding='utf-8') as f:
            f.write(example['text'])

        if (i + 1) % 100000 == 0:
            print(f"  Saved {i+1:,}/{len(dataset):,} stories")

    print(f"\nDone! Saved {len(dataset):,} stories to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
