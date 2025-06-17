#!/usr/bin/env python3
"""
🚀 OPTIMIZED TRAIN - Fast GPT Model Training with Overfitting Prevention
======================================================================
Train a GPT language model optimized for preventing overfitting and fast training
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

def train_model_optimized(args):
    """Train the model with optimizations to prevent overfitting."""
    print("🚀 OPTIMIZED GPT TRAINING")
    print("=" * 35)
    
    # Detect best available device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Metal Performance Shaders")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Read training data (same as original)
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
                text += file_content + "\n\n"
                print(f"  ✅ {txt_file}: {len(file_content):,} characters")
        
        print(f"✅ Loaded {len(text):,} total characters from {len(txt_files)} files")
    else:
        print(f"❌ Error: Data path not found: {args.data}")
        sys.exit(1)
    
    # Create save directory
    os.makedirs(args.output, exist_ok=True)
    
    # Build tokenizer with optimized vocab size for semantic understanding
    print("🔤 Building optimized tokenizer...")
    tokenizer = BPETokenizer()
    # Use full vocab size to ensure proper word-level tokenization
    effective_vocab_size = args.vocab_size
    tokenizer.train(text, max_vocab_size=effective_vocab_size)
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
    
    # Create optimized config to prevent overfitting
    config = Config(
        vocab_size=vocab_size,
        n_embd=args.embedding_dim,
        n_head=args.num_heads,
        n_layer=args.num_layers,
        block_size=args.context_size,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        dropout=args.dropout,  # Higher dropout to prevent overfitting
        weight_decay=args.weight_decay,  # L2 regularization
        grad_clip=1.0,  # Gradient clipping
        fp16=args.mixed_precision and device != "cpu",  # Mixed precision for speed
    )
    
    print(f"🏗️ Model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H on {device}")
    print(f"⚙️ Training: dropout={config.dropout}, weight_decay={config.weight_decay}")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create datasets with better validation split (80/20 instead of 90/10)
    print("🔄 Creating train/validation split...")
    
    # Use 80/20 split for better validation
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_dataset = TextDataset(text=train_text, tokenizer=tokenizer, block_size=config.block_size)
    val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, block_size=config.block_size)
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=args.output
    )
    
    # Implement early stopping
    early_stop_patience = args.early_stopping_patience
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    print(f"🏃 Starting optimized training with early stopping (patience={early_stop_patience})...")
    print(f"🛡️ Overfitting prevention: dropout={args.dropout}, weight_decay={args.weight_decay}")
    start_time = time.time()
    
    try:
        # Enhanced training loop with monitoring
        epoch = 0
        training_interrupted = False
        
        for epoch in range(config.max_epochs):
            # Train one epoch
            epoch_start = time.time()
            avg_train_loss = trainer.train_epoch()
            
            # Evaluate
            eval_results = trainer.evaluate()
            val_loss = eval_results if isinstance(eval_results, float) else eval_results.get('val_loss', float('inf'))
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                trainer.save_checkpoint(Path(args.output) / "best_model.pth")
                print(f"📈 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Best! | Time: {epoch_time:.1f}s | Total: {total_time:.1f}s")
            else:
                early_stop_counter += 1
                print(f"📊 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Patience: {early_stop_counter}/{early_stop_patience} | Time: {epoch_time:.1f}s | Total: {total_time:.1f}s")
                
                if early_stop_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs")
                    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
                    break
            
            # Regular checkpoint saving
            if (epoch + 1) % config.save_interval == 0:
                trainer.save_checkpoint()
                
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user at epoch {epoch+1}")
        training_interrupted = True
    
    training_time = time.time() - start_time
    
    if not training_interrupted:
        print(f"✅ Training completed in {training_time:.1f}s after {epoch+1} epochs")
    else:
        print(f"⏹️ Training stopped in {training_time:.1f}s after {epoch+1} epochs")
    
    print(f"💾 Model saved to: {args.output}")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    
    # Test generation with better parameters
    print("\n🧪 Testing generation...")
    test_prompt = "Delphine and Beau"
    generated = trainer.generate_text(test_prompt, tokenizer, max_tokens=50, temperature=0.8)
    print(f"📝 Test: {generated}")
    
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train an optimized GPT model with overfitting prevention")
    
    # Required arguments
    parser.add_argument("data", help="Path to training text file or directory containing .txt files")
    parser.add_argument("-o", "--output", default="optimized_model", help="Output directory (default: optimized_model)")
    
    # Model architecture - defaults optimized for semantic understanding
    parser.add_argument("--vocab-size", type=int, default=8000, help="Max vocabulary size (default: 8000)")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers (default: 3)")
    parser.add_argument("--context-size", type=int, default=128, help="Context window size (default: 128)")
    
    # Training parameters - optimized for stability
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    
    # Regularization parameters - prevent overfitting
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience (default: 8)")
    
    # Performance parameters
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval in epochs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint interval in epochs (default: 10)")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training for speed")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data path not found: {args.data}")
        sys.exit(1)
    
    print(f"🎯 Training with regularization: dropout={args.dropout}, weight_decay={args.weight_decay}")
    print(f"⏰ Early stopping patience: {args.early_stopping_patience} epochs")
    
    best_val_loss = train_model_optimized(args)
    
    if best_val_loss < 5.0:  # Reasonable threshold
        print("\n🎉 Training completed successfully with good validation loss!")
    else:
        print(f"\n⚠️ Training completed but validation loss ({best_val_loss:.4f}) is high.")
        print("   Consider: reducing model size, increasing dropout, or getting more diverse data")

if __name__ == "__main__":
    main()
