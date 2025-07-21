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
    from src.tokenizer import DefaultTokenizer as BPETokenizer
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
        print(f"⚠️  Tokenizer not found at {tokenizer_path}")
        # Try to find a tokenizer in existing model directories
        fallback_tokenizers = [
            "better_fine_tuned/tokenizer.json",
            "fine_tuned_small_model/tokenizer.json"
        ]
        tokenizer_found = False
        for fallback in fallback_tokenizers:
            if os.path.exists(fallback):
                tokenizer_path = fallback
                print(f"✅ Using fallback tokenizer from {fallback}")
                tokenizer_found = True
                break
        
        if not tokenizer_found:
            print(f"❌ Error: No tokenizer found. Please ensure a tokenizer.json exists.")
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
    
    # Create config with a larger block size for fine-tuning
    if 'config' in checkpoint:
        # Use saved config but increase block size for better Q&A training
        saved_config = checkpoint['config']
        config = Config(
            vocab_size=len(tokenizer.vocab),
            n_embd=saved_config.get('n_embd', 192),
            n_head=saved_config.get('n_head', 6),
            n_layer=saved_config.get('n_layer', 4),
            block_size=256,  # Increased from 96 to 256 for better Q&A training
            device='cpu'
        )
    else:
        # Default config with larger block size
        config = Config(
            vocab_size=len(tokenizer.vocab),
            n_embd=192,
            n_head=6,
            n_layer=4,
            block_size=256,  # Increased from 96 to 256
            device='cpu'
        )
    
    # Create model with new larger block size
    model = create_model_factory(config, len(tokenizer.vocab))
    
    # Load model state - handle potential size mismatch for block_size change
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle size mismatches when changing block_size
    # Get old block size from saved config, default to 96 for older models
    if 'config' in checkpoint:
        old_block_size = checkpoint['config'].get('block_size', 96)
    else:
        old_block_size = 96
    new_block_size = config.block_size
    
    if old_block_size != new_block_size:
        print(f"⚠️  Adapting model from block_size {old_block_size} to {new_block_size}")
        
        # Fix position_ids
        if 'position_ids' in state_dict:
            del state_dict['position_ids']  # Will be recreated automatically
        
        # Fix position embeddings
        if 'position_embedding.weight' in state_dict:
            old_pos_emb = state_dict['position_embedding.weight']
            new_pos_emb = torch.randn(new_block_size, old_pos_emb.shape[1]) * 0.02
            if old_block_size < new_block_size:
                # Copy old embeddings and pad with new ones
                new_pos_emb[:old_block_size] = old_pos_emb
            else:
                # Truncate old embeddings
                new_pos_emb = old_pos_emb[:new_block_size]
            state_dict['position_embedding.weight'] = new_pos_emb
        
        # Fix causal masks for all attention blocks
        keys_to_remove = []
        for key in state_dict.keys():
            if 'attention.causal_mask' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del state_dict[key]  # Will be recreated automatically
    
    # Load the modified state dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"✅ Model loaded. Missing keys (will use default initialization): {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys ignored: {len(unexpected_keys)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
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
    
    # Create fine-tuning dataset with default Alpaca template
    dataset = InstructionDataset(
        args.data, 
        tokenizer, 
        config.block_size
        # Using default templates:
        # instruction_template="### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        # response_template="{output}"
    )
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Fine-tune
    print("🚀 Starting fine-tuning...")
    start_time = time.time()
    
    # Use more conservative fine-tuning settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Add learning rate scheduler (cosine annealing like base training)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    model.train()
    total_loss = 0
    
    # Calculate effective batch size for better reporting
    effective_batches_per_epoch = max(1, len(dataset) // args.batch_size)
    print(f"📊 Training with {effective_batches_per_epoch} batches per epoch (batch_size={args.batch_size})")
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(dataset), args.batch_size):
            batch_data = []
            for j in range(i, min(i + args.batch_size, len(dataset))):
                batch_data.append(dataset[j])
            
            if not batch_data:
                continue
            
            # Prepare batch using the dataset's collate function
            batch = dataset.collate_fn(batch_data)
            
            input_ids = batch['input_ids'][:, :config.block_size]
            target_ids = batch['target_ids'][:, :config.block_size]
            labels_mask = batch['labels_mask'][:, :config.block_size]
            
            # Forward pass
            logits, _ = model(input_ids, target_ids)
            
            # Calculate loss only on response tokens (using labels_mask)
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            mask_flat = labels_mask.view(-1).bool()
            
            # Only calculate loss on masked (response) tokens
            if mask_flat.sum() > 0:
                loss = F.cross_entropy(
                    logits_flat[mask_flat], 
                    targets_flat[mask_flat]
                )
            else:
                loss = torch.tensor(0.0, requires_grad=True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (like base training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        total_loss += avg_loss
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")
    
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
    try:
        shutil.copy(tokenizer_path, os.path.join(args.output, "tokenizer.json"))
        print(f"✅ Tokenizer copied to output directory")
    except FileNotFoundError:
        print(f"⚠️  Warning: Tokenizer not found at {tokenizer_path}")
        # Try to find a tokenizer in existing model directories
        fallback_tokenizers = [
            "better_fine_tuned/tokenizer.json",
            "fine_tuned_small_model/tokenizer.json"
        ]
        tokenizer_copied = False
        for fallback in fallback_tokenizers:
            if os.path.exists(fallback):
                shutil.copy(fallback, os.path.join(args.output, "tokenizer.json"))
                print(f"✅ Using fallback tokenizer from {fallback}")
                tokenizer_copied = True
                break
        
        if not tokenizer_copied:
            print(f"❌ Error: No tokenizer available. Please ensure a tokenizer.json exists in the model directory.")
            sys.exit(1)
    
    print(f"💾 Fine-tuned model saved to: {args.output}")
    
    # Test the fine-tuned model
    print("\n🧪 Testing fine-tuned model...")
    model.eval()
    
    test_questions = ["Who is Beau?", "Where does the story take place?"]
    
    for question in test_questions:
        # Use Alpaca format with instruction/input structure
        prompt = f"### Instruction:\nAnswer the following question about the story.\n\n### Input:\n{question}\n\n### Response:\n"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        
        # Track generated tokens separately
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(50):
                if input_ids.shape[1] >= config.block_size:
                    break
                
                logits, _ = model(input_ids)
                logits = logits[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
                
                # Stop at natural breaking points
                decoded_token = tokenizer.decode([next_token])
                if decoded_token in ['.', '!', '?'] and len(generated_tokens) > 3:
                    break
                elif decoded_token == '\n' and len(generated_tokens) > 5:
                    break
        
        # Decode only the generated part
        if generated_tokens:
            response = tokenizer.decode(generated_tokens).strip()
            # Clean up any formatting artifacts
            if '\n' in response:
                response = response.split('\n')[0].strip()
            print(f"Q: {question}")
            print(f"A: {response}")
        else:
            print(f"Q: {question}")
            print(f"A: [No response generated]")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a trained GPT model")
    
    # Required arguments
    parser.add_argument("model", help="Path to trained model directory")
    parser.add_argument("data", help="Path to Q&A dataset JSON file")
    parser.add_argument("-o", "--output", default="fine_tuned_model", help="Output directory (default: fine_tuned_model)")
    
    # Fine-tuning parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    
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
