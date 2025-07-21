#!/usr/bin/env python3
"""
Fast training using existing cache - Skip the slow cache building!
"""

import sys
import os
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import create_model_factory
from src.training import Trainer
from src.tokenizer import BPETokenizer
from src.data.fast_cache_dataset import StreamingFastDataset

def fast_train(model_name="openwebtext_fast", cache_dir="openwebtext_only/data_cache"):
    """Train using existing cache - no waiting!"""
    
    print("⚡ FAST TRAINING - Using Existing Cache")
    print("=" * 50)
    
    # Check cache exists
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache not found at {cache_dir}")
        print("Available caches:")
        for d in Path(".").glob("*/data_cache"):
            print(f"  - {d}")
        return
        
    # Load tokenizer
    tokenizer_path = cache_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return
        
    print(f"✅ Found cache at {cache_dir}")
    print(f"✅ Found tokenizer at {tokenizer_path}")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
    tokenizer.merges = {}
    for merge_key, idx in tokenizer_data['merges'].items():
        p1, p2 = merge_key.split(',')
        tokenizer.merges[(int(p1), int(p2))] = idx
    
    vocab_size = len(tokenizer.vocab)
    print(f"✅ Loaded tokenizer with {vocab_size} tokens")
    
    # Create fast dataset
    print("⚡ Creating fast streaming dataset...")
    train_dataset = StreamingFastDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        block_size=512,  # Larger context
        cache_size=10000  # Keep 10k sequences in memory
    )
    
    # Create config for larger model
    config = Config(
        vocab_size=vocab_size,
        n_embd=512,
        n_head=8,
        n_layer=12,
        block_size=512,
        batch_size=16,
        learning_rate=3e-4,
        max_epochs=50,
        device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"🏗️ Creating model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create trainer
    os.makedirs(model_name, exist_ok=True)
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        save_dir=model_name
    )
    
    print("🚀 Starting training immediately!")
    print("=" * 50)
    
    # Train
    trainer.train()
    
    print(f"✅ Training complete! Model saved to {model_name}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast training with cached data")
    parser.add_argument("--cache-dir", default="openwebtext_only/data_cache", help="Cache directory")
    parser.add_argument("--model-name", default="openwebtext_fast", help="Output model name")
    
    args = parser.parse_args()
    fast_train(args.model_name, args.cache_dir)