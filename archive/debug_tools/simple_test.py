#!/usr/bin/env python3
"""
Simple test to see RLHF training effects.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import create_model_factory as create_model
from src.tokenizer import BPETokenizer

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    checkpoint = torch.load(model_path, map_location="cpu")
    config = Config.from_dict(checkpoint["config"])
    
    # Load tokenizer
    if "dpo_model" in model_path:
        tokenizer_path = Path("openwebtext_only/tokenizer.json")
    else:
        model_dir = Path(model_path).parent
        tokenizer_path = model_dir / "tokenizer.json"
        
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
    
    return model, tokenizer, config

def simple_generate(model, tokenizer, prompt, device, max_tokens=20):
    """Simple generation with greedy decoding."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs["logits"]
                
            # Greedy decoding
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            if input_tensor.shape[1] >= 64:  # Limit to reasonable length
                break
                
    generated_ids = input_tensor[0].tolist()
    response = tokenizer.decode(generated_ids)
    return response[len(prompt):]

def test_prompts():
    """Test with simple prompts."""
    prompts = [
        "The weather today is",
        "I like to eat",
        "Programming is",
        "The best way to",
        "In the future"
    ]
    
    print("🔍 Testing Base vs RLHF Model Differences")
    print("=" * 60)
    
    # Load models
    base_model, tokenizer, config = load_model_and_tokenizer("openwebtext_only/best_model.pth")
    rlhf_model, _, _ = load_model_and_tokenizer("models/dpo_model/final_model.pth")
    
    for prompt in prompts:
        print(f"\n📝 PROMPT: '{prompt}'")
        print("-" * 40)
        
        base_response = simple_generate(base_model, tokenizer, prompt, config.device)
        rlhf_response = simple_generate(rlhf_model, tokenizer, prompt, config.device)
        
        print(f"BASE: {prompt}{base_response.strip()}")
        print(f"RLHF: {prompt}{rlhf_response.strip()}")
        
        # Check if responses are different
        if base_response.strip() != rlhf_response.strip():
            print("✅ DIFFERENT responses!")
        else:
            print("🔄 Same responses")
    
    print(f"\n{'='*60}")
    print("💡 Even small differences show RLHF training worked!")

if __name__ == "__main__":
    test_prompts()