#!/usr/bin/env python3
"""
❓ ASK - Ask Questions to Trained GPT Model
===========================================
Interactive Q&A with your trained model
"""

import sys
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.config import Config
    from src.model import create_model_factory
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

def generate_response(model, tokenizer, config, prompt, max_tokens=100, temperature=0.7, debug=False):
    """Generate a response to the prompt."""
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    
    if debug:
        print(f"🔍 Debug - Original prompt: '{prompt}'")
        print(f"🔍 Debug - Prompt tokens: {len(tokens)} tokens")
        print(f"🔍 Debug - Decoded tokens: '{tokenizer.decode(tokens)}'")
    
    # Ensure we don't exceed context window
    max_prompt_length = config.block_size - max_tokens
    if len(tokens) > max_prompt_length:
        tokens = tokens[-max_prompt_length:]
        if debug:
            print(f"🔍 Debug - Truncated to {len(tokens)} tokens")
    
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Generate
    generated_tokens = []
    with torch.no_grad():
        for step in range(max_tokens):
            if input_ids.shape[1] >= config.block_size:
                break
                
            logits, _ = model(input_ids)
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            if temperature < 0.1:
                # Greedy sampling for very low temperature
                next_token = torch.argmax(logits).item()
            else:
                next_token = torch.multinomial(probs, 1).item()
            
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            
            # Stop at natural breaking points
            decoded_token = tokenizer.decode([next_token])
            
            if debug and step < 10:  # Debug first 10 tokens
                print(f"🔍 Step {step}: token={next_token}, decoded='{decoded_token}'")
            
            # For Alpaca format, stop at natural sentence breaks
            if decoded_token in ['.', '!', '?'] and len(generated_tokens) > 3:
                break
            elif decoded_token == '\n' and len(generated_tokens) > 5:
                break
            # Stop if we hit another instruction marker (next question)
            elif '###' in decoded_token and len(generated_tokens) > 5:
                break
    
    # Decode only the generated part
    if generated_tokens:
        response = tokenizer.decode(generated_tokens).strip()
        
        if debug:
            print(f"🔍 Debug - Generated tokens: {generated_tokens[:10]}...")
            print(f"🔍 Debug - Raw response: '{response}'")
        
        # Clean up the response
        # Remove any trailing instruction markers
        for end_marker in ['###', '\n\n']:
            if end_marker in response:
                response = response.split(end_marker)[0].strip()
        
        return response
    
    return ""

def load_model(model_dir):
    """Load model and tokenizer from directory."""
    print(f"📂 Loading model from {model_dir}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"❌ Error: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)
    
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} tokens")
    
    # Find model checkpoint
    checkpoint_files = ["fine_tuned_model.pth", "checkpoint_step_1000.pth", "best_model.pth"]
    checkpoint_path = None
    
    for filename in checkpoint_files:
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break
    
    if not checkpoint_path:
        print(f"❌ Error: No model checkpoint found in {model_dir}")
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
    
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {param_count:,} parameters")
    print(f"✅ Architecture: {config.n_layer}L-{config.n_embd}D-{config.n_head}H")
    
    return model, tokenizer, config

def interactive_mode(model, tokenizer, config):
    """Run interactive Q&A session."""
    print("\n❓ INTERACTIVE Q&A MODE")
    print("=" * 25)
    print("Commands:")
    print("  'q: <question>' - Ask a question")
    print("  'story: <prompt>' - Generate story text")
    print("  'temp: <0.1-2.0>' - Change temperature")
    print("  'help' - Show this help")
    print("  'quit' - Exit")
    print("=" * 25)
    
    temperature = 0.7
    
    while True:
        try:
            prompt = input(f"\n🎯 [temp={temperature}] Enter prompt: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if prompt.lower() == 'help':
                print("\nAvailable commands:")
                print("  q: Who is Beau?")
                print("  q: Where does the story take place?")
                print("  story: Delphine and Beau walked through")
                print("  temp: 0.8")
                continue
                
            if prompt.startswith('temp:'):
                try:
                    new_temp = float(prompt.split(':', 1)[1].strip())
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"✅ Temperature set to {temperature}")
                    else:
                        print("❌ Temperature must be between 0.1 and 2.0")
                except ValueError:
                    print("❌ Invalid temperature format")
                continue
            
            if prompt.startswith('debug:'):
                debug_mode = prompt[6:].strip().lower() in ['on', 'true', '1']
                print(f"✅ Debug mode {'enabled' if debug_mode else 'disabled'}")
                continue
            
            # Process prompt
            if prompt.startswith('q:') or prompt.startswith('q '):
                # Question format - use Alpaca template format with the same instruction from training data
                question = prompt[2:].strip()
                full_prompt = f"### Instruction:\nYou are knowledgeable about the story of Beau, a puppy living in New Orleans with his owner Madame Delphine. You can help answer questions about Beau's adventures, the characters in his story, and the places they visit in the French Quarter and beyond.\n\n### Input:\n{question}\n\n### Response:\n"
                max_tokens = 50  
                debug_mode = True  
                print(f"🤖 Answering question...")
            else:
                # Story mode (remove 'story:' prefix if present)
                if prompt.startswith('story:'):
                    prompt = prompt[6:].strip()
                # Use simple story continuation format
                full_prompt = f"{prompt}"
                max_tokens = 80
                debug_mode = False
                print(f"🤖 Generating story...")
            
            # Generate response
            try:
                response = generate_response(model, tokenizer, config, full_prompt, max_tokens, temperature, debug=debug_mode)
                print(f"📝 {response}")
            except Exception as e:
                print(f"❌ Generation error: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_questions(model, tokenizer, config, questions, temperature=0.7):
    """Answer a list of questions."""
    print(f"\n❓ ANSWERING {len(questions)} QUESTIONS")
    print("=" * 30)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        
        # Use the exact instruction from training data
        prompt = f"### Instruction:\nYou are knowledgeable about the story of Beau, a puppy living in New Orleans with his owner Madame Delphine. You can help answer questions about Beau's adventures, the characters in his story, and the places they visit in the French Quarter and beyond.\n\n### Input:\n{question}\n\n### Response:\n"
        
        try:
            response = generate_response(model, tokenizer, config, prompt, max_tokens=50, temperature=temperature, debug=True)
            print(f"A{i}: {response}")
        except Exception as e:
            print(f"A{i}: ❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ask questions to your trained GPT model")
    
    # Required arguments
    parser.add_argument("model", help="Path to trained model directory")
    
    # Optional arguments
    parser.add_argument("-q", "--question", help="Ask a single question")
    parser.add_argument("-f", "--file", help="File containing questions (one per line)")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for generation (default: 0.7)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model directory not found: {args.model}")
        sys.exit(1)
    
    # Load model
    model, tokenizer, config = load_model(args.model)
    
    # Handle different modes
    if args.question:
        # Single question mode
        print(f"\n❓ SINGLE QUESTION MODE")
        print("=" * 25)
        batch_questions(model, tokenizer, config, [args.question], args.temperature)
    
    elif args.file:
        # File mode
        if not os.path.exists(args.file):
            print(f"❌ Error: Questions file not found: {args.file}")
            sys.exit(1)
        
        with open(args.file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        batch_questions(model, tokenizer, config, questions, args.temperature)
    
    else:
        # Interactive mode (default)
        interactive_mode(model, tokenizer, config)

if __name__ == "__main__":
    main()
