#!/usr/bin/env python3
"""
🎯 FINE-TUNE - Fine-tune a trained GPT model
============================================
Fine-tune a pre-trained model on specific tasks
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.config import Config
    from src.model import create_model_factory
    from src.data import InstructionDataset
    from src.training import FineTuningTrainer
    from src.tokenizer import BPETokenizer
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_tokenizer(tokenizer_path):
    """Load tokenizer from JSON file."""
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = BPETokenizer()
    
    # Reconstruct vocab
    vocab = {int(k): bytes.fromhex(v) for k, v in data['vocab'].items()}
    
    # Reconstruct merges
    merges = {}
    for merge_str, idx in data['merges'].items():
        p1, p2 = merge_str.split(',')
        merges[(int(p1), int(p2))] = idx
    
    tokenizer.vocab = vocab
    tokenizer.merges = merges
    
    return tokenizer

def fine_tune_model(args):
    """Fine-tune the model."""
    print("🎯 FINE-TUNING GPT MODEL")
    print("=" * 25)
    
    # Load base model
    print(f"📂 Loading base model from {args.model}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.model, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} tokens")
    
    # Load model checkpoint
    checkpoint_files = ["checkpoint_step_1000.pth", "best_model.pth", "fine_tuned_model.pth"]
    checkpoint_path = None
    
    for filename in checkpoint_files:
        candidate = os.path.join(args.model, filename)
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break
    
    if not checkpoint_path:
        print(f"❌ Error: No model checkpoint found in {args.model}")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✅ Checkpoint loaded: {os.path.basename(checkpoint_path)}")
    
    # Create config
    if 'config' in checkpoint:
        # Use saved config if available
        saved_config = checkpoint['config']
        config = Config(
            vocab_size=len(tokenizer.vocab),
            n_embd=saved_config.get('n_embd', 192),
            n_head=saved_config.get('n_head', 6),
            n_layer=saved_config.get('n_layer', 4),
            block_size=saved_config.get('block_size', 96),
            device='cpu'
        )
    else:
        # Default config for older checkpoints
        config = Config(
            vocab_size=len(tokenizer.vocab),
            n_embd=192,
            n_head=6,
            n_layer=4,
            block_size=96,
            device='cpu'
        )
    
    # Create model
    model = create_model_factory(config, len(tokenizer.vocab))
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {param_count:,} parameters")
    
    # Load Q&A data
    print(f"📚 Loading Q&A data from {args.data}")
    with open(args.data, 'r') as f:
        qa_data = json.load(f)
    
    if isinstance(qa_data, list):
        qa_pairs = qa_data
    elif 'data' in qa_data:
        qa_pairs = qa_data['data']
    else:
        qa_pairs = qa_data
    
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs")
    
    # Create fine-tuning dataset
    dataset = InstructionDataset(args.data, tokenizer, config.block_size)
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Fine-tune
    print("🚀 Starting fine-tuning...")
    start_time = time.time()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    total_loss = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(dataset), args.batch_size):
            batch_data = []
            for j in range(i, min(i + args.batch_size, len(dataset))):
                batch_data.append(dataset[j])
            
            if not batch_data:
                continue
            
            # Prepare batch
            max_len = max(len(item['input_ids']) for item in batch_data)
            input_ids = []
            targets = []
            
            for item in batch_data:
                ids = item['input_ids']
                # Convert tensor to list if needed and pad to max length
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                padded = ids + [0] * (max_len - len(ids))
                input_ids.append(padded[:config.block_size])
                targets.append(padded[:config.block_size])
            
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
            
            # Forward pass
            logits, loss = model(input_ids, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        total_loss += avg_loss
        print(f"Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.4f}")
    
    final_loss = total_loss / args.epochs
    training_time = time.time() - start_time
    
    print(f"✅ Fine-tuning complete! Final loss: {final_loss:.4f}")
    print(f"⏱️  Fine-tuning time: {training_time:.1f}s")
    
    # Save fine-tuned model
    model_path = os.path.join(args.output, "fine_tuned_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size
        },
        'final_loss': final_loss,
        'training_time': training_time
    }, model_path)
    
    # Copy tokenizer
    import shutil
    shutil.copy(tokenizer_path, os.path.join(args.output, "tokenizer.json"))
    
    print(f"💾 Fine-tuned model saved to: {args.output}")
    
    # Test the fine-tuned model
    print("\n🧪 Testing fine-tuned model...")
    model.eval()
    
    test_questions = ["Who is Beau?", "Where does the story take place?"]
    
    for question in test_questions:
        prompt = f"Q: {question}\nA:"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(50):
                if input_ids.shape[1] >= config.block_size:
                    break
                
                logits, _ = model(input_ids)
                logits = logits[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
                
                if tokenizer.decode([next_token]) in ['\n', '?']:
                    break
        
        response = tokenizer.decode(input_ids[0].tolist())
        if "A:" in response:
            answer = response.split("A:")[-1].strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
            print(f"Q: {question}")
            print(f"A: {answer}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a trained GPT model")
    
    # Required arguments
    parser.add_argument("model", help="Path to trained model directory")
    parser.add_argument("data", help="Path to Q&A dataset JSON file")
    parser.add_argument("-o", "--output", default="fine_tuned_model", help="Output directory (default: fine_tuned_model)")
    
    # Fine-tuning parameters
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model directory not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    fine_tune_model(args)
    print("\n🎉 Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
