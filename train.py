#!/usr/bin/env python3
"""
🚀 TRAIN - Production GPT Model Training
========================================
Train a production-quality GPT language model
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.config import Config
    from src.model import create_model_factory
    from src.data import TextDataset
    from src.training import Trainer
    from src.tokenizer import BPETokenizer
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def train_model(args):
    """Train the production model."""
    print("🚀 PRODUCTION GPT TRAINING")
    print("=" * 30)
    
    # Read training data
    print(f"📖 Reading data from {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✅ Loaded {len(text):,} characters")
    
    # Create save directory
    os.makedirs(args.output, exist_ok=True)
    
    # Build tokenizer
    print("🔤 Building tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(text, max_vocab_size=args.vocab_size)
    vocab_size = len(tokenizer.vocab)
    print(f"✅ Tokenizer built: {vocab_size} tokens")
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output, "tokenizer.json")
    vocab_hex = {str(k): v.hex() for k, v in tokenizer.vocab.items()}
    merges_str = {f"{p1},{p2}": idx for (p1, p2), idx in tokenizer.merges.items()}
    
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'vocab': vocab_hex,
            'merges': merges_str
        }, f, indent=2)
    print(f"✅ Tokenizer saved: {tokenizer_path}")
    
    # Create config
    config = Config(
        vocab_size=vocab_size,
        n_embd=args.embedding_dim,
        n_head=args.num_heads,
        n_layer=args.num_layers,
        block_size=args.context_size,
        max_iters=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device='cpu'
    )
    
    print(f"🏗️ Model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create dataset
    dataset = TextDataset(text=text, tokenizer=tokenizer, block_size=config.block_size)
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=dataset,
        save_dir=args.output
    )
    
    # Train
    print("🏃 Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"✅ Training complete in {training_time:.1f}s")
    print(f"💾 Model saved to: {args.output}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    test_prompt = "Delphine and Beau"
    generated = trainer.generate_text(test_prompt, tokenizer, max_tokens=50, temperature=0.7)
    print(f"📝 Test: {generated}")

def main():
    parser = argparse.ArgumentParser(description="Train a production GPT model")
    
    # Required arguments
    parser.add_argument("data", help="Path to training text file")
    parser.add_argument("-o", "--output", default="model", help="Output directory (default: model)")
    
    # Model architecture
    parser.add_argument("--vocab-size", type=int, default=1500, help="Vocabulary size (default: 1500)")
    parser.add_argument("--embedding-dim", type=int, default=192, help="Embedding dimension (default: 192)")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--context-size", type=int, default=96, help="Context window size (default: 96)")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=300, help="Training iterations (default: 300)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    train_model(args)
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
