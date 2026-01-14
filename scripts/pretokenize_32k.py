#!/usr/bin/env python3
"""Pre-tokenize data with 32k HuggingFace tokenizer."""

import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer_hf import HFTokenizer

# Global tokenizer for multiprocessing
_tokenizer = None


def init_worker(tokenizer_path: str):
    """Initialize tokenizer in each worker."""
    global _tokenizer
    _tokenizer = HFTokenizer(tokenizer_path)


def encode_file(story_file: Path) -> list:
    """Encode a single story file with BOS/EOS."""
    global _tokenizer
    try:
        with open(story_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if text:
            tokens = [_tokenizer.BOS_TOKEN_ID]
            tokens.extend(_tokenizer.encode(text))
            tokens.append(_tokenizer.EOS_TOKEN_ID)
            return tokens
    except Exception as e:
        print(f"Error: {e}")
    return []


def pretokenize(data_dir: Path, tokenizer_path: Path, output_dir: Path,
                train_ratio: float = 0.9, num_workers: int = None):
    """Tokenize all stories in parallel."""

    if num_workers is None:
        num_workers = cpu_count()

    # Get files
    story_files = sorted(data_dir.glob("story_*.txt"))
    total = len(story_files)
    print(f"Found {total:,} files in {data_dir}")

    # Split
    split_idx = int(total * train_ratio)
    train_files = story_files[:split_idx]
    val_files = story_files[split_idx:]
    print(f"Train: {len(train_files):,}, Val: {len(val_files):,}")
    print(f"Using {num_workers} workers")

    # Encode training data
    print("\nEncoding training data...")
    start = time.time()

    with Pool(num_workers, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        train_tokens = []
        chunk_size = 10000
        for i in range(0, len(train_files), chunk_size):
            chunk = train_files[i:i + chunk_size]
            results = pool.map(encode_file, chunk)
            for tokens in results:
                train_tokens.extend(tokens)
            processed = min(i + chunk_size, len(train_files))
            elapsed = time.time() - start
            rate = processed / elapsed
            remaining = (len(train_files) - processed) / rate if rate > 0 else 0
            print(f"  {processed:,}/{len(train_files):,} ({rate:.0f}/s, ~{remaining:.0f}s left)")

    train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    train_time = time.time() - start
    print(f"Train tokens: {len(train_tokens):,} in {train_time:.1f}s")

    # Encode validation data
    print("\nEncoding validation data...")
    start = time.time()

    with Pool(num_workers, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        results = pool.map(encode_file, val_files)
        val_tokens = []
        for tokens in results:
            val_tokens.extend(tokens)

    val_tokens = torch.tensor(val_tokens, dtype=torch.long)
    val_time = time.time() - start
    print(f"Val tokens: {len(val_tokens):,} in {val_time:.1f}s")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_tokens, output_dir / "train_tokens.pt")
    torch.save(val_tokens, output_dir / "val_tokens.pt")
    print(f"\nSaved to {output_dir}")
    print(f"Total time: {train_time + val_time:.1f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/tinystories_500k")
    parser.add_argument("--tokenizer", default="models/tokenizer_32k")
    parser.add_argument("--output", default="data/tokenized_500k_32k")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    pretokenize(
        data_dir=PROJECT_ROOT / args.data_dir,
        tokenizer_path=PROJECT_ROOT / args.tokenizer,
        output_dir=PROJECT_ROOT / args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
