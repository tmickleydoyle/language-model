#!/usr/bin/env python3
"""
RLHF Data Management Utility

This script provides utilities for managing, validating, and organizing
RLHF preference datasets.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def validate_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Validate a preference dataset and return statistics."""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    stats = {
        "total_examples": len(data),
        "categories": {},
        "quality_types": {},
        "sources": {},
        "avg_chosen_length": 0,
        "avg_rejected_length": 0,
        "valid_examples": 0,
        "invalid_examples": []
    }
    
    chosen_lengths = []
    rejected_lengths = []
    
    for i, example in enumerate(data):
        # Check required fields
        required_fields = ["prompt", "chosen", "rejected"]
        missing_fields = [field for field in required_fields if field not in example]
        
        if missing_fields:
            stats["invalid_examples"].append({
                "index": i,
                "missing_fields": missing_fields
            })
            continue
            
        stats["valid_examples"] += 1
        
        # Calculate lengths
        chosen_length = len(example["chosen"].split())
        rejected_length = len(example["rejected"].split())
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        
        # Extract metadata
        metadata = example.get("metadata", {})
        
        category = metadata.get("category", "unknown")
        stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        quality_type = metadata.get("quality", "unknown")
        stats["quality_types"][quality_type] = stats["quality_types"].get(quality_type, 0) + 1
        
        source = metadata.get("source", "unknown")
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
    
    # Calculate averages
    if chosen_lengths:
        stats["avg_chosen_length"] = sum(chosen_lengths) / len(chosen_lengths)
        stats["avg_rejected_length"] = sum(rejected_lengths) / len(rejected_lengths)
    
    return stats

def merge_datasets(dataset_paths: List[Path], output_path: Path) -> None:
    """Merge multiple datasets into one."""
    
    merged_data = []
    source_info = {}
    
    for path in dataset_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Add source info to metadata
        source_name = path.stem
        for example in data:
            if "metadata" not in example:
                example["metadata"] = {}
            example["metadata"]["original_source"] = source_name
            
        merged_data.extend(data)
        source_info[source_name] = len(data)
    
    # Save merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✅ Merged {len(merged_data)} examples from {len(dataset_paths)} datasets")
    print("📊 Source breakdown:")
    for source, count in source_info.items():
        print(f"   {source}: {count} examples")

def split_dataset(dataset_path: Path, train_ratio: float = 0.8) -> None:
    """Split a dataset into train/validation sets."""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle data
    import random
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save splits
    base_path = dataset_path.parent / dataset_path.stem
    train_path = f"{base_path}_train.json"
    val_path = f"{base_path}_val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ Split dataset:")
    print(f"   Train: {len(train_data)} examples → {train_path}")
    print(f"   Val: {len(val_data)} examples → {val_path}")

def filter_dataset(dataset_path: Path, filters: Dict[str, Any], output_path: Path) -> None:
    """Filter dataset based on criteria."""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    filtered_data = []
    
    for example in data:
        include = True
        metadata = example.get("metadata", {})
        
        # Apply filters
        for key, value in filters.items():
            if key == "min_chosen_length":
                if len(example["chosen"].split()) < value:
                    include = False
                    break
            elif key == "max_chosen_length":
                if len(example["chosen"].split()) > value:
                    include = False
                    break
            elif key == "category":
                if metadata.get("category") != value:
                    include = False
                    break
            elif key == "source":
                if metadata.get("source") != value:
                    include = False
                    break
        
        if include:
            filtered_data.append(example)
    
    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✅ Filtered dataset: {len(data)} → {len(filtered_data)} examples")
    print(f"📁 Saved to: {output_path}")

def list_datasets() -> None:
    """List all available datasets in the rlhf directory."""
    
    rlhf_dir = Path(__file__).parent
    
    print("📊 Available RLHF Datasets:")
    print("=" * 50)
    
    for category in ["curated", "synthetic", "combined"]:
        category_dir = rlhf_dir / "datasets" / category
        if category_dir.exists():
            print(f"\n📁 {category.upper()}:")
            for dataset_file in category_dir.glob("*.json"):
                stats = validate_dataset(dataset_file)
                print(f"   {dataset_file.name}:")
                print(f"      Examples: {stats['total_examples']}")
                print(f"      Valid: {stats['valid_examples']}")
                if stats['invalid_examples']:
                    print(f"      Invalid: {len(stats['invalid_examples'])}")

def main():
    parser = argparse.ArgumentParser(description="Manage RLHF datasets")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all datasets")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset")
    validate_parser.add_argument("dataset", type=str, help="Path to dataset file")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    merge_parser.add_argument("datasets", nargs="+", help="Paths to dataset files")
    merge_parser.add_argument("-o", "--output", required=True, help="Output file path")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val")
    split_parser.add_argument("dataset", type=str, help="Path to dataset file")
    split_parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    
    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter dataset")
    filter_parser.add_argument("dataset", type=str, help="Path to dataset file")
    filter_parser.add_argument("-o", "--output", required=True, help="Output file path")
    filter_parser.add_argument("--category", type=str, help="Filter by category")
    filter_parser.add_argument("--source", type=str, help="Filter by source")
    filter_parser.add_argument("--min-length", type=int, help="Minimum chosen response length")
    filter_parser.add_argument("--max-length", type=int, help="Maximum chosen response length")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_datasets()
        
    elif args.command == "validate":
        dataset_path = Path(args.dataset)
        stats = validate_dataset(dataset_path)
        
        print(f"📊 Dataset Validation: {dataset_path.name}")
        print("=" * 50)
        print(f"Total examples: {stats['total_examples']}")
        print(f"Valid examples: {stats['valid_examples']}")
        
        if stats['invalid_examples']:
            print(f"Invalid examples: {len(stats['invalid_examples'])}")
            for invalid in stats['invalid_examples'][:5]:  # Show first 5
                print(f"   Index {invalid['index']}: Missing {invalid['missing_fields']}")
        
        print(f"\nAverage lengths:")
        print(f"   Chosen: {stats['avg_chosen_length']:.1f} words")
        print(f"   Rejected: {stats['avg_rejected_length']:.1f} words")
        
        if stats['categories']:
            print(f"\nCategories:")
            for cat, count in stats['categories'].items():
                print(f"   {cat}: {count}")
                
        if stats['quality_types']:
            print(f"\nQuality types:")
            for qt, count in stats['quality_types'].items():
                print(f"   {qt}: {count}")
        
    elif args.command == "merge":
        dataset_paths = [Path(p) for p in args.datasets]
        output_path = Path(args.output)
        merge_datasets(dataset_paths, output_path)
        
    elif args.command == "split":
        dataset_path = Path(args.dataset)
        split_dataset(dataset_path, args.ratio)
        
    elif args.command == "filter":
        dataset_path = Path(args.dataset)
        output_path = Path(args.output)
        
        filters = {}
        if args.category:
            filters["category"] = args.category
        if args.source:
            filters["source"] = args.source
        if args.min_length:
            filters["min_chosen_length"] = args.min_length
        if args.max_length:
            filters["max_chosen_length"] = args.max_length
            
        filter_dataset(dataset_path, filters, output_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()