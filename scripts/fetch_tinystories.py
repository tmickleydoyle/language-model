#!/usr/bin/env python3
"""Fetch TinyStories dataset for training small language models.

TinyStories is a dataset of short stories specifically designed for
training small language models. It contains simple, coherent narrative
prose that works well with models under 50M parameters.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Usage:
    python scripts/fetch_tinystories.py --output data/tinystories
    python scripts/fetch_tinystories.py --output data/tinystories --max-stories 10000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Generator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed."""
    try:
        from datasets import load_dataset
        return True
    except ImportError:
        logger.error("The 'datasets' package is required. Install with:")
        logger.error("  pip install datasets")
        return False


def fetch_tinystories(
    output_dir: str,
    max_stories: Optional[int] = None,
    min_length: int = 100,
    max_length: int = 2000,
    split: str = "train",
    skip: int = 0,
) -> dict:
    """Fetch and save TinyStories dataset.

    Args:
        output_dir: Directory to save stories
        max_stories: Maximum number of stories to fetch (None for all)
        min_length: Minimum story length in characters
        max_length: Maximum story length in characters
        split: Dataset split to use ('train' or 'validation')
        skip: Number of stories to skip from the beginning

    Returns:
        Statistics dictionary
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TinyStories dataset from HuggingFace...")
    logger.info("This may take a few minutes on first run...")

    # Load the dataset
    try:
        dataset = load_dataset("roneneldan/TinyStories", split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure you have internet connection and datasets package installed")
        raise

    logger.info(f"Dataset loaded: {len(dataset)} stories in {split} split")

    stats = {
        "total_available": len(dataset),
        "fetched": 0,
        "skipped_short": 0,
        "skipped_long": 0,
        "skipped_offset": skip,
        "total_chars": 0,
    }

    # Process stories
    stories = []
    for i, item in enumerate(dataset):
        # Skip first N stories if requested
        if i < skip:
            continue

        if max_stories and stats["fetched"] >= max_stories:
            break

        text = item.get("text", "").strip()

        # Filter by length
        if len(text) < min_length:
            stats["skipped_short"] += 1
            continue
        if len(text) > max_length:
            stats["skipped_long"] += 1
            continue

        stories.append(text)
        stats["fetched"] += 1
        stats["total_chars"] += len(text)

        # Progress logging
        if stats["fetched"] % 1000 == 0:
            logger.info(f"Processed {stats['fetched']} stories...")

    # Save as individual files (for file-aware training)
    logger.info(f"Saving {len(stories)} stories to {output_dir}...")
    for i, story in enumerate(stories):
        file_path = output_path / f"story_{i:06d}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(story)

    # Also save combined file for convenience
    combined_path = output_path / "_combined.txt"
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("\n\n---\n\n".join(stories))

    stats["combined_file"] = str(combined_path)
    stats["estimated_tokens"] = stats["total_chars"] // 4

    return stats


def create_train_val_split(
    input_dir: str,
    train_dir: str,
    val_dir: str,
    val_ratio: float = 0.1,
) -> dict:
    """Split stories into train and validation sets.

    Args:
        input_dir: Directory with story files
        train_dir: Output directory for training stories
        val_dir: Output directory for validation stories
        val_ratio: Fraction for validation

    Returns:
        Statistics dictionary
    """
    import random

    input_path = Path(input_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # Get all story files
    story_files = sorted(input_path.glob("story_*.txt"))
    random.shuffle(story_files)

    # Split
    val_count = int(len(story_files) * val_ratio)
    val_files = story_files[:val_count]
    train_files = story_files[val_count:]

    # Copy files
    import shutil
    for f in train_files:
        shutil.copy(f, train_path / f.name)
    for f in val_files:
        shutil.copy(f, val_path / f.name)

    return {
        "total": len(story_files),
        "train": len(train_files),
        "val": len(val_files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch TinyStories dataset for small LM training"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/tinystories",
        help="Output directory for stories"
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Maximum number of stories to fetch (default: all)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum story length in characters"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2000,
        help="Maximum story length in characters"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of stories to skip from the beginning"
    )
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Also create train/val split directories"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return 1

    print("=" * 60)
    print("TinyStories Dataset Fetcher")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"Max stories: {args.max_stories or 'all'}")
    print(f"Skip first: {args.skip}")
    print(f"Length filter: {args.min_length}-{args.max_length} chars")

    # Fetch stories
    try:
        stats = fetch_tinystories(
            output_dir=args.output,
            max_stories=args.max_stories,
            min_length=args.min_length,
            max_length=args.max_length,
            split=args.split,
            skip=args.skip,
        )
    except Exception as e:
        logger.error(f"Failed to fetch stories: {e}")
        return 1

    print(f"\nResults:")
    print(f"  Available: {stats['total_available']:,}")
    print(f"  Fetched: {stats['fetched']:,}")
    print(f"  Skipped (too short): {stats['skipped_short']:,}")
    print(f"  Skipped (too long): {stats['skipped_long']:,}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  Est. tokens: {stats['estimated_tokens']:,}")
    print(f"  Combined file: {stats['combined_file']}")

    # Optionally create train/val split
    if args.create_split:
        print("\nCreating train/val split...")
        split_stats = create_train_val_split(
            args.output,
            f"{args.output}_train",
            f"{args.output}_val",
        )
        print(f"  Train: {split_stats['train']:,} stories")
        print(f"  Val: {split_stats['val']:,} stories")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
