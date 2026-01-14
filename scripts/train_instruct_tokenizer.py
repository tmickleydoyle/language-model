#!/usr/bin/env python3
"""Train a tokenizer on instruction data using HuggingFace tokenizers."""

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "instruct_1m" / "data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "tokenizer_instruct"


def format_example(instruction: str, input_text: str, output: str) -> str:
    """Format an example in instruction format."""
    text = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        text += f"### Input:\n{input_text}\n\n"
    text += f"### Response:\n{output}"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    print(f"Training tokenizer with vocab size {args.vocab_size}...")

    # Create temporary file with formatted text for training
    temp_file = PROJECT_ROOT / "data" / "instruct_1m" / "formatted_for_tokenizer.txt"
    print(f"Formatting data for tokenizer training...")

    with open(DATA_FILE, 'r', encoding='utf-8') as f_in, \
         open(temp_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            example = json.loads(line)
            formatted = format_example(
                example.get('instruction', ''),
                example.get('input', ''),
                example.get('output', '')
            )
            f_out.write(formatted + '\n\n')

            if (i + 1) % 100_000 == 0:
                print(f"  Formatted {i+1:,} examples")

    print(f"Formatted text saved to {temp_file}")

    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Define special tokens
    special_tokens = [
        "<pad>",    # 0
        "<unk>",    # 1
        "<bos>",    # 2
        "<eos>",    # 3
    ]

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=args.min_frequency,
    )

    # Train
    print(f"\nTraining tokenizer on {temp_file}...")
    tokenizer.train([str(temp_file)], trainer)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    print(f"\nTokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Test tokenization
    print("\nTesting tokenization...")
    test_text = "### Instruction:\nSummarize this text.\n\n### Input:\nHello world.\n\n### Response:\nA greeting."
    encoded = tokenizer.encode(test_text)
    print(f"Test text: {test_text[:100]}...")
    print(f"Tokens: {encoded.ids[:20]}...")
    print(f"Decoded: {tokenizer.decode(encoded.ids)[:100]}...")

    # Clean up temp file
    temp_file.unlink()
    print(f"\nCleaned up temp file")


if __name__ == "__main__":
    main()
