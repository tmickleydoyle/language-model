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
    if os.path.isfile(args.data):
        print(f"📖 Reading data from file {args.data}")
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Loaded {len(text):,} characters from single file")
    elif os.path.isdir(args.data):
        print(f"📖 Reading data from directory {args.data}")
        text = ""
        txt_files = sorted([f for f in os.listdir(args.data) if f.endswith('.txt')])
        
        if not txt_files:
            print(f"❌ Error: No .txt files found in directory: {args.data}")
            sys.exit(1)
        
        print(f"📚 Found {len(txt_files)} text files:")
        for txt_file in txt_files:
            file_path = os.path.join(args.data, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                text += file_content + "\n\n"  # Add spacing between files
                print(f"  ✅ {txt_file}: {len(file_content):,} characters")
        
        print(f"✅ Loaded {len(text):,} total characters from {len(txt_files)} files")
    else:
        print(f"❌ Error: Data path not found: {args.data}")
        sys.exit(1)
    
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
    
    # Create config with optimized parameters to prevent overfitting
    config = Config(
        vocab_size=vocab_size,
        n_embd=args.embedding_dim,
        n_head=args.num_heads,
        n_layer=args.num_layers,
        block_size=args.context_size,
        max_iters=args.iterations,
        max_epochs=args.epochs,  # Use command line argument
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,  # Use command line argument for device
        save_interval=args.save_interval,  # Use command line argument
        eval_interval=args.eval_interval,  # Use command line argument
        dropout=args.dropout,  # Use command line argument
        weight_decay=args.weight_decay,  # Use command line argument
        grad_clip=1.0,  # Add gradient clipping
        fp16=args.mixed_precision,  # Use mixed precision if requested
    )
    
    print(f"🏗️ Model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create datasets with larger validation split for better generalization
    print("🔄 Creating train/validation split...")
    
    # Split text into train (80%) and validation (20%) for better validation
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = TextDataset(text=train_text, tokenizer=tokenizer, block_size=config.block_size)
    val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, block_size=config.block_size)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # Create trainer with early stopping monitoring
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=args.output
    )
    
    # Add early stopping mechanism
    def early_stopping_callback(trainer_instance, val_loss, patience=5):
        """Simple early stopping implementation"""
        if not hasattr(trainer_instance, 'early_stop_counter'):
            trainer_instance.early_stop_counter = 0
            trainer_instance.best_val_loss_early = float('inf')
        
        if val_loss < trainer_instance.best_val_loss_early:
            trainer_instance.best_val_loss_early = val_loss
            trainer_instance.early_stop_counter = 0
        else:
            trainer_instance.early_stop_counter += 1
            
        if trainer_instance.early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            return True
        return False
    
    # Train with monitoring
    print("🏃 Starting training with overfitting prevention...")
    start_time = time.time()
    
    # Override the training loop to add early stopping
    original_train = trainer.train
    def train_with_early_stopping():
        try:
            return original_train()
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            return None
    
    trainer.train = train_with_early_stopping
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
    parser.add_argument("data", help="Path to training text file or directory containing .txt files")
    parser.add_argument("-o", "--output", default="model", help="Output directory (default: model)")
    
    # Model architecture
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size (default: 8000)")
    parser.add_argument("--embedding-dim", type=int, default=192, help="Embedding dimension (default: 192)")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--context-size", type=int, default=256, help="Context window size (default: 256)")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=300, help="Training iterations (default: 300)")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4, reduced for stability)")
    
    # Regularization parameters
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for regularization (default: 0.1)")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience (default: 5)")
    
    # Performance parameters
    parser.add_argument("--eval-interval", type=int, default=25, help="Evaluation interval (default: 25)")
    parser.add_argument("--save-interval", type=int, default=50, help="Save checkpoint interval (default: 50)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: cpu, cuda, mps, or auto (default: auto)")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training for speed")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data path not found: {args.data}")
        sys.exit(1)
    
    train_model(args)
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
