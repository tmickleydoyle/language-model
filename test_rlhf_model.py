#!/usr/bin/env python3
"""
Test RLHF-trained model with custom questions.

This script allows you to interactively test your model with any question.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import create_model_factory as create_model
from src.tokenizer import BPETokenizer

def generate_response(model, tokenizer, prompt, device="cpu", max_length=100, temperature=0.8, top_k=50):
    """Generate a response from the model."""
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs["logits"]
                
            # Apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')
                
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            # Stop if we hit max length for model  
            if hasattr(model, 'config') and input_tensor.shape[1] >= model.config.block_size:
                break
                
    # Decode
    generated_ids = input_tensor[0].tolist()
    response = tokenizer.decode(generated_ids)
    
    # Return only the generated part
    return response[len(prompt):]

def compare_models(base_model_path, rlhf_model_path, prompt):
    """Compare base model vs RLHF model responses."""
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    
    # Load models
    checkpoint = torch.load(base_model_path, map_location="cpu")
    config = Config.from_dict(checkpoint["config"])
    
    # Load tokenizer
    model_dir = Path(base_model_path).parent
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = BPETokenizer()
    
    # Load tokenizer data
    import json
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
    tokenizer.merges = {}
    for merge_key, idx in tokenizer_data['merges'].items():
        p1, p2 = merge_key.split(',')
        tokenizer.merges[(int(p1), int(p2))] = idx
    
    # Build encoder/decoder
    tokenizer.encoder = {}
    tokenizer.decoder = {}
    for token_id, token_bytes in tokenizer.vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        tokenizer.encoder[token_str] = token_id
        tokenizer.decoder[token_id] = token_str
    
    # Load base model
    print("\n📚 BASE MODEL RESPONSE:")
    print("-" * 40)
    base_model = create_model(config, config.vocab_size)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model = base_model.to(config.device)
    base_model.eval()
    
    base_response = generate_response(base_model, tokenizer, prompt, config.device)
    print(base_response.strip())
    
    # Load RLHF model
    print("\n🎯 RLHF MODEL RESPONSE:")
    print("-" * 40)
    rlhf_checkpoint = torch.load(rlhf_model_path, map_location=config.device)
    rlhf_model = create_model(config, config.vocab_size)
    rlhf_model.load_state_dict(rlhf_checkpoint["model_state_dict"])
    rlhf_model = rlhf_model.to(config.device)
    rlhf_model.eval()
    
    rlhf_response = generate_response(rlhf_model, tokenizer, prompt, config.device)
    print(rlhf_response.strip())
    
    print(f"\n{'='*60}\n")

def interactive_test(model_path):
    """Interactive testing mode."""
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    config = Config.from_dict(checkpoint["config"])
    
    # Load tokenizer
    if "dpo_model" in model_path:
        # For RLHF model, use base model's tokenizer
        tokenizer_path = Path("openwebtext_only/tokenizer.json")
    else:
        model_dir = Path(model_path).parent
        tokenizer_path = model_dir / "tokenizer.json"
        
    tokenizer = BPETokenizer()
    
    # Load tokenizer data
    import json
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
    tokenizer.merges = {}
    for merge_key, idx in tokenizer_data['merges'].items():
        p1, p2 = merge_key.split(',')
        tokenizer.merges[(int(p1), int(p2))] = idx
    
    # Build encoder/decoder
    tokenizer.encoder = {}
    tokenizer.decoder = {}
    for token_id, token_bytes in tokenizer.vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        tokenizer.encoder[token_str] = token_id
        tokenizer.decoder[token_id] = token_str
    
    # Load model
    model = create_model(config, config.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()
    
    print("🤖 RLHF Model Test Interface")
    print("Type 'quit' to exit, 'compare' to compare with base model")
    print("-" * 50)
    
    while True:
        prompt = input("\n📝 Enter your prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
            
        if prompt.lower() == 'compare':
            test_prompt = input("Enter prompt to compare: ").strip()
            compare_models(
                "openwebtext_only/best_model.pth",
                "models/dpo_model/final_model.pth",
                test_prompt
            )
            continue
            
        if not prompt:
            continue
            
        print("\n🤔 Generating response...")
        response = generate_response(model, tokenizer, prompt, config.device)
        print(f"\n💬 Response: {response.strip()}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test RLHF model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/dpo_model/final_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base vs RLHF model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to test"
    )
    
    args = parser.parse_args()
    
    if args.compare and args.prompt:
        compare_models(
            "openwebtext_only/best_model.pth",
            args.model,
            args.prompt
        )
    elif args.prompt:
        # Single prompt test
        checkpoint = torch.load(args.model, map_location="cpu")
        config = Config.from_dict(checkpoint["config"])
        
        # Load tokenizer
        tokenizer_path = Path("openwebtext_only/tokenizer.json")
        tokenizer = BPETokenizer()
        
        import json
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
        tokenizer.merges = {}
        for merge_key, idx in tokenizer_data['merges'].items():
            p1, p2 = merge_key.split(',')
            tokenizer.merges[(int(p1), int(p2))] = idx
        
        tokenizer.encoder = {}
        tokenizer.decoder = {}
        for token_id, token_bytes in tokenizer.vocab.items():
            token_str = token_bytes.decode('utf-8', errors='replace')
            tokenizer.encoder[token_str] = token_id
            tokenizer.decoder[token_id] = token_str
        
        # Load model
        model = create_model(config, config.vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)
        model.eval()
        
        response = generate_response(model, tokenizer, args.prompt, config.device)
        print(f"Response: {response.strip()}")
    else:
        # Interactive mode
        interactive_test(args.model)

if __name__ == "__main__":
    main()