#!/usr/bin/env python3
"""Download and sample instruction dataset from HuggingFace."""

import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "instruct_1m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Downloading simple-instruct-dataset from HuggingFace...")
    print(f"Target: {args.num_examples:,} examples")

    # Load dataset (streaming to avoid loading all 35M into memory)
    dataset = load_dataset(
        "javi22/simple-instruct-dataset",
        split="train",
        streaming=True
    )

    # Sample examples
    print(f"\nSampling {args.num_examples:,} examples...")
    random.seed(args.seed)

    # Use reservoir sampling for streaming dataset
    examples = []
    for i, example in enumerate(dataset):
        if i < args.num_examples:
            examples.append(example)
        else:
            # Reservoir sampling
            j = random.randint(0, i)
            if j < args.num_examples:
                examples[j] = example

        if (i + 1) % 1_000_000 == 0:
            print(f"  Processed {i+1:,} examples...")

        # Stop after processing enough for good sampling
        if i >= args.num_examples * 5:
            break

    print(f"Sampled {len(examples):,} examples")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    output_file = OUTPUT_DIR / "data.jsonl"
    print(f"\nSaving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples):
            # Extract only the columns we need
            data = {
                'instruction': example.get('instruction', ''),
                'input': example.get('input', ''),
                'output': example.get('output', '')
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

            if (i + 1) % 100_000 == 0:
                print(f"  Saved {i+1:,}/{len(examples):,}")

    # Print stats
    print(f"\nDone! Saved {len(examples):,} examples to {output_file}")

    # Show some examples
    print("\nSample examples:")
    for i in range(min(3, len(examples))):
        ex = examples[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {ex.get('instruction', '')[:100]}...")
        print(f"Input: {ex.get('input', '')[:100]}...")
        print(f"Output: {ex.get('output', '')[:100]}...")


if __name__ == "__main__":
    main()
