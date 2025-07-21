#!/usr/bin/env python3
"""
Train GPT model using existing cached data from wiki_model/data_cache/batches/
"""

import sys
import os
import json
import pickle
import time
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import create_model_factory
from src.training import Trainer
from src.tokenizer import DefaultTokenizer as BPETokenizer

class CachedDataset:
    """Simple dataset that works with cached batch files"""
    
    def __init__(self, batch_files, tokenizer, block_size=128, max_samples=10000):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache = []
        
        print(f"Loading data from {len(batch_files)} batch files...")
        
        # Load texts from batch files
        all_texts = []
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_texts = pickle.load(f)
                    all_texts.extend(batch_texts)
                    print(f"Loaded {len(batch_texts)} texts from {batch_file.name}")
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
                continue
        
        print(f"Processing {len(all_texts)} texts into training samples...")
        
        # Convert texts to training samples
        current_tokens = []
        for i, text in enumerate(all_texts):
            if len(self.cache) >= max_samples:
                break
                
            try:
                # Tokenize text
                tokens = tokenizer.encode(text)
                current_tokens.extend(tokens)
                
                # Create samples from tokens
                while len(current_tokens) >= block_size + 1 and len(self.cache) < max_samples:
                    sequence = current_tokens[:block_size + 1]
                    current_tokens = current_tokens[block_size:]
                    
                    # Convert to tensors
                    x = torch.tensor(sequence[:-1], dtype=torch.long)
                    y = torch.tensor(sequence[1:], dtype=torch.long)
                    
                    self.cache.append({'input_ids': x, 'labels': y})
                    
                    if len(self.cache) % 1000 == 0:
                        print(f"Processed {len(self.cache)}/{max_samples} samples...")
                        
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                continue
        
        print(f"✅ Created dataset with {len(self.cache)} samples")
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        return self.cache[idx]
    
    def get_batch(self, batch_size, device):
        """Get a batch of data for training"""
        import random
        
        # Randomly sample from cache
        batch_indices = random.sample(range(len(self.cache)), min(batch_size, len(self.cache)))
        
        # Stack inputs and labels
        x_batch = torch.stack([self.cache[i]['input_ids'] for i in batch_indices])
        y_batch = torch.stack([self.cache[i]['labels'] for i in batch_indices])
        
        # Move to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        return x_batch, y_batch

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GPT model on cached data")
    
    # Data arguments
    parser.add_argument("--cache-dir", default="wiki_model/data_cache", 
                       help="Directory containing cached data (default: wiki_model/data_cache)")
    parser.add_argument("--tokenizer-path", default="wiki_model/tokenizer.json",
                       help="Path to tokenizer file (default: wiki_model/tokenizer.json)")
    parser.add_argument("--output-dir", default="wiki_model",
                       help="Output directory for model (default: wiki_model)")
    
    # Model architecture
    parser.add_argument("--embedding-dim", type=int, default=128, 
                       help="Embedding dimension (default: 128)")
    parser.add_argument("--num-heads", type=int, default=4, 
                       help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=3, 
                       help="Number of transformer layers (default: 3)")
    parser.add_argument("--context-size", type=int, default=128, 
                       help="Context window size (default: 128)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Maximum training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=5e-4, 
                       help="Learning rate (default: 5e-4)")
    parser.add_argument("--train-samples", type=int, default=6400, 
                       help="Number of training samples (default: 6400)")
    parser.add_argument("--val-samples", type=int, default=1600, 
                       help="Number of validation samples (default: 1600)")
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.3, 
                       help="Dropout rate (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, 
                       help="Weight decay (default: 0.01)")
    parser.add_argument("--patience", type=int, default=5, 
                       help="Early stopping patience (default: 5)")
    
    # Other options
    parser.add_argument("--save-interval", type=int, default=5, 
                       help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--mixed-precision", action="store_true", 
                       help="Use mixed precision training")
    parser.add_argument("--test-generation", action="store_true", default=True,
                       help="Test text generation after training (default: True)")
    
    args = parser.parse_args()
    
    print("🚀 CACHED DATA TRAINING")
    print("=" * 30)
    print(f"📁 Cache directory: {args.cache_dir}")
    print(f"🔤 Tokenizer: {args.tokenizer_path}")
    print(f"💾 Output: {args.output_dir}")
    
    # Check for existing cached data
    cache_dir = Path(args.cache_dir)
    batch_dir = cache_dir / "batches"
    
    if not batch_dir.exists():
        print("❌ No cached data found. Please run data collection first.")
        sys.exit(1)
    
    batch_files = sorted(batch_dir.glob("batch_*.pkl"))
    if not batch_files:
        print("❌ No batch files found in cache directory")
        sys.exit(1)
    
    print(f"📦 Found {len(batch_files)} batch files")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Metal Performance Shaders")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Load existing tokenizer if available
    if os.path.exists(args.tokenizer_path):
        print("🔤 Loading existing tokenizer...")
        with open(args.tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Reconstruct tokenizer
        tokenizer = BPETokenizer()
        tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
        tokenizer.merges = {tuple(k.split(',')): v for k, v in tokenizer_data['merges'].items()}
        vocab_size = len(tokenizer.vocab)
        print(f"✅ Loaded tokenizer with {vocab_size} tokens")
    else:
        print(f"❌ No tokenizer found at {args.tokenizer_path}. Please run tokenizer training first.")
        sys.exit(1)
    
    # Create model config
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
        eval_interval=1,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        fp16=args.mixed_precision and device != "cpu",
    )
    
    print(f"🏗️ Model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H on {device}")
    print(f"📊 Training: {args.train_samples} samples, Validation: {args.val_samples} samples")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create datasets from cached data
    print("📊 Creating datasets from cached data...")
    train_dataset = CachedDataset(batch_files, tokenizer, config.block_size, args.train_samples)
    val_dataset = CachedDataset(batch_files, tokenizer, config.block_size, args.val_samples)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=args.output_dir
    )
    
    # Training loop
    print(f"🏃 Starting training for {config.max_epochs} epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(config.max_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            avg_train_loss = trainer.train_epoch()
            
            # Evaluate
            eval_results = trainer.evaluate()
            val_loss = eval_results if isinstance(eval_results, float) else eval_results.get('val_loss', float('inf'))
            
            epoch_time = time.time() - epoch_start
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                trainer.save_checkpoint(Path(args.output_dir) / "best_model.pth")
                print(f"📈 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Best! 🎯 | Time: {epoch_time:.1f}s")
            else:
                patience_counter += 1
                print(f"📊 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Patience: {patience_counter}/{args.patience} | Time: {epoch_time:.1f}s")
                
                if patience_counter >= args.patience:
                    print(f"🛑 Early stopping! No improvement for {args.patience} epochs")
                    break
            
            # Save checkpoint
            if (epoch + 1) % config.save_interval == 0:
                trainer.save_checkpoint()
                
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f}s")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    
    # Test generation
    if args.test_generation:
        print("\n🧪 Testing generation...")
        test_prompt = "The future of artificial intelligence"
        generated = trainer.generate_text(test_prompt, tokenizer, max_tokens=50, temperature=0.8)
        print(f"📝 Generated: {generated}")

if __name__ == "__main__":
    main()