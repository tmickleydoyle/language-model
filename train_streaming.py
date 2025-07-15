#!/usr/bin/env python3
"""
🌊 STREAMING TRAIN - Train GPT model with streaming data from APIs
=================================================================
Train without storing large datasets locally - stream from multiple sources
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
    from src.training import Trainer
    from src.tokenizer import BPETokenizer
    from src.data.streaming_dataset import create_streaming_datasets, DATASET_CONFIGS
    from src.data.streaming_data_loader import StreamingDataLoader
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def train_model_streaming(args):
    """Train the model with streaming data from APIs."""
    print("🌊 STREAMING GPT TRAINING")
    print("=" * 30)
    
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
    
    # Get data source configuration
    if args.dataset_config in DATASET_CONFIGS:
        sources = DATASET_CONFIGS[args.dataset_config]
        print(f"📊 Using predefined config: {args.dataset_config}")
    else:
        # Custom configuration
        sources = {"openwebtext": 5000, "wikipedia": 2000}
        print("📊 Using default configuration")
    
    print("🔄 Data sources:")
    total_samples = 0
    for source, count in sources.items():
        print(f"  • {source}: {count:,} samples")
        total_samples += count
    print(f"  📈 Total: {total_samples:,} samples")
    
    # Create save directory
    os.makedirs(args.output, exist_ok=True)
    
    # Build tokenizer from streaming data
    print("🔤 Building tokenizer from streaming data...")
    
    # Get sample text for tokenizer training
    loader = StreamingDataLoader()
    sample_sources = {k: min(v, 500) for k, v in sources.items()}  # Limit for tokenizer
    sample_texts = []
    
    print("📚 Collecting sample texts for tokenizer...")
    for text in loader.stream_mixed_sources(sample_sources, total_samples=2000):
        sample_texts.append(text)
        if len(sample_texts) >= 100:  # Limit sample size
            break
    
    combined_text = "\n\n".join(sample_texts)
    print(f"✅ Collected {len(combined_text):,} characters for tokenizer training")
    
    # Train tokenizer
    tokenizer = BPETokenizer()
    effective_vocab_size = args.vocab_size
    tokenizer.train(combined_text, max_vocab_size=effective_vocab_size)
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
        eval_interval=args.eval_interval,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        fp16=args.mixed_precision and device != "cpu",
    )
    
    print(f"🏗️ Model: {config.n_layer}L-{config.n_embd}D-{config.n_head}H on {device}")
    print(f"⚙️ Streaming: cache_size={args.cache_size}, refresh_rate={args.refresh_rate}")
    
    # Create model
    model = create_model_factory(config, vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {param_count:,} parameters")
    
    # Create streaming datasets
    print("🌊 Creating streaming datasets...")
    train_dataset, val_dataset = create_streaming_datasets(
        sources=sources,
        tokenizer=tokenizer,
        block_size=config.block_size,
        train_split=0.8,
        use_cache=args.use_cache,
        cache_size=args.cache_size
    )
    
    print(f"✅ Train dataset created")
    print(f"✅ Val dataset created")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=args.output
    )
    
    # Training loop with cache refresh
    early_stop_patience = args.early_stopping_patience
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    print(f"🏃 Starting streaming training...")
    print(f"🔄 Cache refresh every {args.refresh_rate} epochs")
    start_time = time.time()
    
    try:
        epoch = 0
        training_interrupted = False
        
        for epoch in range(config.max_epochs):
            # Refresh cache periodically
            if (epoch > 0 and epoch % args.refresh_rate == 0 and 
                args.use_cache and hasattr(train_dataset, 'refresh_cache')):
                print(f"🔄 Refreshing training cache at epoch {epoch+1}...")
                train_dataset.refresh_cache(refresh_ratio=0.3)
                if hasattr(val_dataset, 'refresh_cache'):
                    val_dataset.refresh_cache(refresh_ratio=0.5)
            
            # Train one epoch
            epoch_start = time.time()
            avg_train_loss = trainer.train_epoch()
            
            # Evaluate
            eval_results = trainer.evaluate()
            val_loss = eval_results if isinstance(eval_results, float) else eval_results.get('val_loss', float('inf'))
            
            epoch_time = time.time() - epoch_start
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                trainer.save_checkpoint(Path(args.output) / "best_model.pth")
                print(f"📈 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Best! 🌊 | Time: {epoch_time:.1f}s")
            else:
                early_stop_counter += 1
                print(f"📊 Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Patience: {early_stop_counter}/{early_stop_patience} | Time: {epoch_time:.1f}s")
                
                if early_stop_counter >= early_stop_patience:
                    print(f"🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs")
                    break
            
            # Regular checkpoint saving
            if (epoch + 1) % config.save_interval == 0:
                trainer.save_checkpoint()
                
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user at epoch {epoch+1}")
        training_interrupted = True
    
    training_time = time.time() - start_time
    
    if not training_interrupted:
        print(f"✅ Streaming training completed in {training_time:.1f}s after {epoch+1} epochs")
    else:
        print(f"⏹️ Training stopped in {training_time:.1f}s after {epoch+1} epochs")
    
    print(f"💾 Model saved to: {args.output}")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    test_prompt = "The future of artificial intelligence"
    generated = trainer.generate_text(test_prompt, tokenizer, max_tokens=50, temperature=0.8)
    print(f"📝 Generated: {generated}")
    
    return best_val_loss

def main():
    parser = argparse.ArgumentParser(description="Train GPT model with streaming data from APIs")
    
    # Required arguments
    parser.add_argument("-o", "--output", default="streaming_model", help="Output directory (default: streaming_model)")
    
    # Data source configuration
    parser.add_argument("--dataset-config", default="small_mixed", 
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Predefined dataset configuration (default: small_mixed)")
    
    # Model architecture
    parser.add_argument("--vocab-size", type=int, default=8000, help="Max vocabulary size (default: 8000)")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers (default: 3)")
    parser.add_argument("--context-size", type=int, default=128, help="Context window size (default: 128)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    
    # Streaming parameters
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use cached streaming dataset")
    parser.add_argument("--cache-size", type=int, default=10000, help="Size of data cache (default: 10000)")
    parser.add_argument("--refresh-rate", type=int, default=5, help="Refresh cache every N epochs (default: 5)")
    
    # Regularization parameters
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default: 0.3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience (default: 8)")
    
    # Performance parameters
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval in epochs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint interval in epochs (default: 10)")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training for speed")
    
    args = parser.parse_args()
    
    print("🌊 STREAMING DATA TRAINING")
    print("=" * 30)
    print(f"📊 Dataset config: {args.dataset_config}")
    print(f"💾 Cache size: {args.cache_size:,} samples")
    print(f"🔄 Refresh rate: every {args.refresh_rate} epochs")
    print(f"🎯 Output: {args.output}")
    
    best_val_loss = train_model_streaming(args)
    
    if best_val_loss < 3.0:
        print("\n🎉 Streaming training completed successfully!")
    else:
        print(f"\n⚠️ Training completed but validation loss ({best_val_loss:.4f}) is high.")

if __name__ == "__main__":
    main()