#!/usr/bin/env python3
"""Pre-tokenize TinyStories data with parallel processing and save as .pt files."""

import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import BPETokenizer

# Global tokenizer for multiprocessing (loaded once per worker)
_tokenizer = None
_bos_id = None
_eos_id = None


def init_worker(tokenizer_path: str):
    """Initialize tokenizer in each worker process."""
    global _tokenizer, _bos_id, _eos_id
    _tokenizer = BPETokenizer()
    _tokenizer.load(tokenizer_path)
    _bos_id = _tokenizer.BOS_TOKEN_ID
    _eos_id = _tokenizer.EOS_TOKEN_ID


def encode_file(story_file: Path) -> list:
    """Encode a single story file. Returns list of token IDs with BOS/EOS."""
    global _tokenizer, _bos_id, _eos_id
    try:
        with open(story_file, 'r', encoding='utf-8') as f:
            story_text = f.read().strip()
        if story_text:
            tokens = [_bos_id]
            tokens.extend(_tokenizer.encode(story_text))
            tokens.append(_eos_id)
            return tokens
    except Exception as e:
        print(f"Error encoding {story_file}: {e}")
    return []


def pretokenize(data_dir: Path, tokenizer_path: Path, output_dir: Path,
                train_ratio: float = 0.9, num_workers: int = None):
    """Tokenize all stories in parallel and save as train/val .pt files."""

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    # Get story files
    story_files = sorted(data_dir.glob("story_*.txt"))
    total_files = len(story_files)
    print(f"Found {total_files:,} story files in {data_dir}")

    if total_files == 0:
        raise ValueError(f"No story files found in {data_dir}")

    # Split into train/val
    split_idx = int(total_files * train_ratio)
    train_files = story_files[:split_idx]
    val_files = story_files[split_idx:]
    print(f"Train: {len(train_files):,} stories, Val: {len(val_files):,} stories")
    print(f"Using {num_workers} parallel workers")

    # Encode training data in parallel
    print("\nEncoding training stories...")
    start_time = time.time()

    with Pool(num_workers, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        # Process in chunks to show progress
        chunk_size = 10000
        train_tokens = []

        for i in range(0, len(train_files), chunk_size):
            chunk = train_files[i:i + chunk_size]
            results = pool.map(encode_file, chunk)
            for tokens in results:
                train_tokens.extend(tokens)

            elapsed = time.time() - start_time
            processed = min(i + chunk_size, len(train_files))
            rate = processed / elapsed
            remaining = (len(train_files) - processed) / rate if rate > 0 else 0
            print(f"  {processed:,}/{len(train_files):,} ({rate:.0f} files/s, ~{remaining:.0f}s remaining)")

    train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    train_time = time.time() - start_time
    print(f"Training tokens: {len(train_tokens):,} in {train_time:.1f}s")

    # Encode validation data in parallel
    print("\nEncoding validation stories...")
    start_time = time.time()

    with Pool(num_workers, initializer=init_worker, initargs=(str(tokenizer_path),)) as pool:
        results = pool.map(encode_file, val_files)
        val_tokens = []
        for tokens in results:
            val_tokens.extend(tokens)

    val_tokens = torch.tensor(val_tokens, dtype=torch.long)
    val_time = time.time() - start_time
    print(f"Validation tokens: {len(val_tokens):,} in {val_time:.1f}s")

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_tokens.pt"
    val_path = output_dir / "val_tokens.pt"

    print(f"\nSaving to {output_dir}...")
    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    print(f"Saved: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Saved: {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\nTotal time: {train_time + val_time:.1f}s")
    print("\nDone! Update train_with_muon.py to load these files.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-tokenize data with parallel processing")
    parser.add_argument("--data-dir", type=str, default="data/tinystories_500k",
                        help="Directory containing story files")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer_10k/tokenizer",
                        help="Path to tokenizer (without extension)")
    parser.add_argument("--output", type=str, default="data/tokenized_500k_10k",
                        help="Output directory for .pt files")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Ratio of data for training (default: 0.9)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1)")

    args = parser.parse_args()

    pretokenize(
        data_dir=PROJECT_ROOT / args.data_dir,
        tokenizer_path=PROJECT_ROOT / args.tokenizer,
        output_dir=PROJECT_ROOT / args.output,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
