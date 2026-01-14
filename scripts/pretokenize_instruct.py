#!/usr/bin/env python3
"""Pre-tokenize instruction data with loss masks.

Creates token sequences with masks indicating which tokens should contribute to loss.
Only output tokens (after ### Response:) contribute to loss.
"""

import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer_hf import HFTokenizer

DATA_FILE = PROJECT_ROOT / "data" / "instruct_1m" / "data.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer_instruct"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized_instruct_1m"

# Global tokenizer for multiprocessing
_tokenizer = None


def init_worker(tokenizer_path: str):
    """Initialize tokenizer in each worker."""
    global _tokenizer
    _tokenizer = HFTokenizer(tokenizer_path)


def format_and_tokenize(example: dict) -> tuple:
    """Format example and return tokens with loss mask.

    Returns:
        (tokens, loss_mask) where loss_mask is 1 for output tokens, 0 otherwise
    """
    global _tokenizer

    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    # Format prompt (instruction + input)
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    # Tokenize prompt and output separately
    prompt_tokens = _tokenizer.encode(prompt)
    output_tokens = _tokenizer.encode(output)

    # Combine with BOS/EOS
    tokens = [_tokenizer.BOS_TOKEN_ID] + prompt_tokens + output_tokens + [_tokenizer.EOS_TOKEN_ID]

    # Loss mask: 0 for BOS + prompt, 1 for output + EOS
    # We want to predict the output given the prompt
    loss_mask = [0] * (1 + len(prompt_tokens)) + [1] * (len(output_tokens) + 1)

    return tokens, loss_mask


def process_example(line: str) -> tuple:
    """Process a single example line."""
    try:
        example = json.loads(line.strip())
        return format_and_tokenize(example)
    except Exception as e:
        return None, None


def main():
    print("Pre-tokenizing instruction data with loss masks...")

    # Load examples
    print(f"Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Loaded {len(lines):,} examples")

    # Split into train/val (90/10)
    split_idx = int(len(lines) * 0.9)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    print(f"Train: {len(train_lines):,}, Val: {len(val_lines):,}")

    # Process training data
    print("\nProcessing training data...")
    num_workers = cpu_count()

    with Pool(num_workers, initializer=init_worker, initargs=(str(TOKENIZER_PATH),)) as pool:
        results = []
        chunk_size = 10000
        for i in range(0, len(train_lines), chunk_size):
            chunk = train_lines[i:i + chunk_size]
            chunk_results = pool.map(process_example, chunk)
            results.extend(chunk_results)
            print(f"  Processed {min(i + chunk_size, len(train_lines)):,}/{len(train_lines):,}")

    # Flatten and filter None results
    train_tokens = []
    train_masks = []
    for tokens, mask in results:
        if tokens is not None:
            train_tokens.append(tokens)
            train_masks.append(mask)

    print(f"Training examples: {len(train_tokens):,}")

    # Process validation data
    print("\nProcessing validation data...")
    with Pool(num_workers, initializer=init_worker, initargs=(str(TOKENIZER_PATH),)) as pool:
        results = pool.map(process_example, val_lines)

    val_tokens = []
    val_masks = []
    for tokens, mask in results:
        if tokens is not None:
            val_tokens.append(tokens)
            val_masks.append(mask)

    print(f"Validation examples: {len(val_tokens):,}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {OUTPUT_DIR}...")
    torch.save({
        'tokens': train_tokens,
        'masks': train_masks
    }, OUTPUT_DIR / "train.pt")

    torch.save({
        'tokens': val_tokens,
        'masks': val_masks
    }, OUTPUT_DIR / "val.pt")

    # Stats
    train_total_tokens = sum(len(t) for t in train_tokens)
    val_total_tokens = sum(len(t) for t in val_tokens)
    train_output_tokens = sum(sum(m) for m in train_masks)

    print(f"\nStats:")
    print(f"  Train: {len(train_tokens):,} examples, {train_total_tokens:,} tokens")
    print(f"  Val: {len(val_tokens):,} examples, {val_total_tokens:,} tokens")
    print(f"  Output tokens (for loss): {train_output_tokens:,} ({100*train_output_tokens/train_total_tokens:.1f}%)")


if __name__ == "__main__":
    main()
