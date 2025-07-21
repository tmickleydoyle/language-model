#!/usr/bin/env python3
"""
Estimate tokenization time for different data sizes and tokenizer types.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.tokenizer import DefaultTokenizer, BPETokenizer
    from src.tokenizer import FAST_TOKENIZER_AVAILABLE
    if FAST_TOKENIZER_AVAILABLE:
        from src.tokenizer import FastBPETokenizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def get_sample_text(size_mb: float) -> str:
    """Generate or load sample text of approximately the given size in MB."""
    # Use a diverse sample that's representative of training data
    base_text = """The quick brown fox jumps over the lazy dog. This is a sample text that contains various characters, punctuation, and common English words. It includes numbers like 123, 456, and symbols like @#$%^&*(). The text should be representative of typical training data for language models.

Here's some code: def hello_world(): print("Hello, world!")

And some more varied content: Machine learning is transforming how we process and understand text. Large language models can generate human-like text, answer questions, and perform various natural language tasks. The training process involves processing massive amounts of text data, which requires efficient tokenization algorithms.

Let's add some more diversity: 人工智能 (artificial intelligence), émotions (emotions), and математика (mathematics). Special characters: ñ, ü, ç, ø, and emojis: 🚀 🎯 ⚡ 🔥 💡
"""
    
    # Calculate approximate size and repeat text to reach target size
    base_size_mb = len(base_text.encode('utf-8')) / (1024 * 1024)
    repeat_count = max(1, int(size_mb / base_size_mb))
    
    return base_text * repeat_count


def benchmark_tokenizer(tokenizer_class, text: str, vocab_size: int = 10000) -> Dict[str, float]:
    """Benchmark a tokenizer's performance."""
    print(f"📊 Benchmarking {tokenizer_class.__name__}...")
    
    # Initialize tokenizer
    tokenizer = tokenizer_class()
    
    # Measure training time
    train_start = time.time()
    tokenizer.train(text, max_vocab_size=vocab_size, verbose=False)
    train_time = time.time() - train_start
    
    # Measure encoding time
    encode_start = time.time()
    tokens = tokenizer.encode(text)
    encode_time = time.time() - encode_start
    
    # Calculate throughput
    text_size_mb = len(text.encode('utf-8')) / (1024 * 1024)
    
    return {
        'train_time': train_time,
        'encode_time': encode_time,
        'text_size_mb': text_size_mb,
        'tokens_count': len(tokens),
        'vocab_size': tokenizer.vocab_size,
        'train_throughput_mb_per_sec': text_size_mb / train_time if train_time > 0 else float('inf'),
        'encode_throughput_mb_per_sec': text_size_mb / encode_time if encode_time > 0 else float('inf')
    }


def estimate_time_for_size(benchmark_results: Dict[str, float], target_size_mb: float) -> Dict[str, float]:
    """Estimate tokenization time for a target data size."""
    train_throughput = benchmark_results['train_throughput_mb_per_sec']
    encode_throughput = benchmark_results['encode_throughput_mb_per_sec']
    
    estimated_train_time = target_size_mb / train_throughput if train_throughput > 0 else float('inf')
    estimated_encode_time = target_size_mb / encode_throughput if encode_throughput > 0 else float('inf')
    
    return {
        'estimated_train_time': estimated_train_time,
        'estimated_encode_time': estimated_encode_time,
        'target_size_mb': target_size_mb
    }


def format_time(seconds: float) -> str:
    """Format time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    else:
        return f"{seconds/86400:.2f} days"


def get_data_size_estimates() -> List[Tuple[str, float]]:
    """Get common data size estimates."""
    return [
        ("Small dataset (10 MB)", 10),
        ("Medium dataset (100 MB)", 100),
        ("Large dataset (1 GB)", 1024),
        ("Very large dataset (10 GB)", 10240),
        ("Massive dataset (100 GB)", 102400),
        ("OpenWebText-like (40 GB)", 40960),
        ("The Pile-like (800 GB)", 819200),
    ]


def analyze_existing_data(data_path: str) -> Optional[float]:
    """Analyze existing data to estimate size."""
    try:
        path = Path(data_path)
        if path.is_file():
            size_bytes = path.stat().st_size
            return size_bytes / (1024 * 1024)  # Convert to MB
        elif path.is_dir():
            total_size = 0
            for file_path in path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.txt', '.json', '.jsonl']:
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        else:
            print(f"❌ Path not found: {data_path}")
            return None
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Estimate tokenization time for your data")
    parser.add_argument('--data-path', type=str, help='Path to your data file/directory to analyze')
    parser.add_argument('--benchmark-size', type=float, default=1.0, 
                       help='Size in MB to use for benchmarking (default: 1.0)')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size for tokenizer training (default: 10000)')
    parser.add_argument('--save-results', type=str, help='Save benchmark results to JSON file')
    
    args = parser.parse_args()
    
    print("🚀 Tokenization Time Estimator")
    print("=" * 50)
    
    # Generate benchmark text
    print(f"📝 Generating {args.benchmark_size:.1f} MB of benchmark text...")
    benchmark_text = get_sample_text(args.benchmark_size)
    
    # Available tokenizers
    tokenizers_to_test = [BPETokenizer]
    if FAST_TOKENIZER_AVAILABLE:
        tokenizers_to_test.append(FastBPETokenizer)
    
    # Benchmark each tokenizer
    results = {}
    for tokenizer_class in tokenizers_to_test:
        try:
            results[tokenizer_class.__name__] = benchmark_tokenizer(
                tokenizer_class, benchmark_text, args.vocab_size
            )
        except Exception as e:
            print(f"❌ Failed to benchmark {tokenizer_class.__name__}: {e}")
    
    # Display benchmark results
    print("\n📊 Benchmark Results:")
    print("-" * 50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Training time: {format_time(result['train_time'])}")
        print(f"  Encoding time: {format_time(result['encode_time'])}")
        print(f"  Training throughput: {result['train_throughput_mb_per_sec']:.2f} MB/sec")
        print(f"  Encoding throughput: {result['encode_throughput_mb_per_sec']:.2f} MB/sec")
        print(f"  Tokens produced: {result['tokens_count']:,}")
        print(f"  Vocab size: {result['vocab_size']:,}")
    
    # Analyze user's data if provided
    if args.data_path:
        print(f"\n📁 Analyzing your data: {args.data_path}")
        data_size_mb = analyze_existing_data(args.data_path)
        if data_size_mb:
            print(f"📊 Your data size: {data_size_mb:.2f} MB ({data_size_mb/1024:.2f} GB)")
            
            print("\n⏱️ Estimated tokenization time for your data:")
            print("-" * 50)
            for name, result in results.items():
                estimate = estimate_time_for_size(result, data_size_mb)
                print(f"\n{name}:")
                print(f"  Training time: {format_time(estimate['estimated_train_time'])}")
                print(f"  Encoding time: {format_time(estimate['estimated_encode_time'])}")
                total_time = estimate['estimated_train_time'] + estimate['estimated_encode_time']
                print(f"  Total time: {format_time(total_time)}")
    
    # Show estimates for common data sizes
    print(f"\n📈 Time estimates for common dataset sizes:")
    print("-" * 50)
    
    for dataset_name, size_mb in get_data_size_estimates():
        print(f"\n{dataset_name}:")
        for name, result in results.items():
            estimate = estimate_time_for_size(result, size_mb)
            total_time = estimate['estimated_train_time'] + estimate['estimated_encode_time']
            print(f"  {name}: {format_time(total_time)}")
    
    # Speed comparison
    if len(results) > 1:
        print(f"\n⚡ Speed Comparison:")
        print("-" * 50)
        baseline_name = 'BPETokenizer'
        if baseline_name in results:
            baseline = results[baseline_name]
            for name, result in results.items():
                if name != baseline_name:
                    train_speedup = baseline['train_time'] / result['train_time']
                    encode_speedup = baseline['encode_time'] / result['encode_time']
                    print(f"{name} vs {baseline_name}:")
                    print(f"  Training: {train_speedup:.1f}x faster")
                    print(f"  Encoding: {encode_speedup:.1f}x faster")
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.save_results}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("-" * 50)
    if FAST_TOKENIZER_AVAILABLE:
        print("✅ Fast tokenizer is available and will be used by default")
        print("🚀 For large datasets, the fast tokenizer provides significant speedup")
    else:
        print("⚠️  Fast tokenizer not available. Install with: pip install tokenizers")
        print("🐌 Using custom BPE implementation (much slower for large datasets)")
    
    if args.data_path:
        data_size_mb = analyze_existing_data(args.data_path)
        if data_size_mb and data_size_mb > 1000:  # > 1GB
            print("📊 Your dataset is large. Consider:")
            print("   - Using streaming data loading to avoid memory issues")
            print("   - Processing data in chunks if tokenization takes too long")
            print("   - Using a smaller vocabulary size for faster training")


if __name__ == "__main__":
    main()