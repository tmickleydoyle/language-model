#!/usr/bin/env python3
"""
Model Evaluation and Comparison Tool
===================================
Compare multiple models on various tasks and metrics
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.config import Config
    from src.model import create_model_factory
    from src.tokenizer import BPETokenizer
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_model_and_tokenizer(model_path):
    """Load a model and its tokenizer."""
    try:
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = BPETokenizer()
        
        # Reconstruct vocab
        vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
        
        # Reconstruct merges
        merges = {}
        for merge_str, idx in tokenizer_data['merges'].items():
            p1, p2 = merge_str.split(',')
            merges[(int(p1), int(p2))] = idx
        
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        
        # Reconstruct encoder and decoder for compatibility
        tokenizer.encoder = {}
        tokenizer.decoder = {}
        for token_id, token_bytes in vocab.items():
            try:
                token_str = token_bytes.decode('utf-8')
                tokenizer.encoder[token_str] = token_id
                tokenizer.decoder[token_id] = token_str
            except UnicodeDecodeError:
                # Handle non-UTF8 bytes as raw byte representation
                token_str = f"<byte_{token_id}>"
                tokenizer.encoder[token_str] = token_id  
                tokenizer.decoder[token_id] = token_str
        
        # Load model
        checkpoint_files = ["checkpoint_step_1000.pth", "best_model.pth", "fine_tuned_model.pth"]
        checkpoint_path = None
        
        for filename in checkpoint_files:
            candidate = os.path.join(model_path, filename)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
        
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create config
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            config = Config(
                vocab_size=len(tokenizer.vocab),
                n_embd=saved_config.get('n_embd', 192),
                n_head=saved_config.get('n_head', 6),
                n_layer=saved_config.get('n_layer', 4),
                block_size=saved_config.get('block_size', 256),
                device='cpu'
            )
        else:
            config = Config(
                vocab_size=len(tokenizer.vocab),
                n_embd=192,
                n_head=6,
                n_layer=4,
                block_size=256,
                device='cpu'
            )
        
        # Create and load model
        model = create_model_factory(config, len(tokenizer.vocab))
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        return model, tokenizer, config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def generate_response(model, tokenizer, config, prompt, max_tokens=50, temperature=0.7):
    """Generate a response from the model."""
    try:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if input_ids.shape[1] >= config.block_size:
                    break
                
                logits, _ = model(input_ids)
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
                
                # Stop on certain tokens
                if tokenizer.decode([next_token]) in ['\n', '?', '!', '.']:
                    break
        
        response = tokenizer.decode(input_ids[0].tolist())
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"

def evaluate_model_on_tasks(model, tokenizer, config, model_name):
    """Evaluate a model on various tasks."""
    print(f"\n📊 Evaluating {model_name}")
    print("=" * 50)
    
    # Task 1: Story Questions
    print("\n🔍 Task 1: Story-specific Questions")
    story_questions = [
        "What is the name of the puppy in the story?",
        "What type of animal is Beau?",
        "Where do Beau and Madame Delphine live?",
        "Who is Beau's owner?"
    ]
    
    story_scores = []
    for question in story_questions:
        prompt = f"### Instruction:\nAnswer the question about the story.\n\n### Input:\n{question}\n\n### Response:\n"
        response = generate_response(model, tokenizer, config, prompt)
        
        # Extract answer
        if "### Response:\n" in response:
            answer = response.split("### Response:\n")[-1].strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
        else:
            answer = response.strip()
        
        # Simple scoring based on expected keywords
        score = 0
        if "beau" in answer.lower():
            score += 1
        if any(word in answer.lower() for word in ["puppy", "dog", "animal"]):
            score += 1
        if any(word in answer.lower() for word in ["new orleans", "royal street", "delphine"]):
            score += 1
            
        story_scores.append(score)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Score: {score}/3\n")
    
    # Task 2: General Instructions
    print("🎯 Task 2: General Instruction Following")
    general_questions = [
        "How do you train a puppy to sit?",
        "What makes a good pet owner?",
        "What should I consider when adopting a puppy?",
        "Describe what New Orleans is known for."
    ]
    
    instruction_scores = []
    for question in general_questions:
        prompt = f"### Instruction:\nProvide helpful information.\n\n### Input:\n{question}\n\n### Response:\n"
        response = generate_response(model, tokenizer, config, prompt, max_tokens=80)
        
        if "### Response:\n" in response:
            answer = response.split("### Response:\n")[-1].strip()
            if '\n' in answer:
                answer = answer.split('\n')[0].strip()
        else:
            answer = response.strip()
        
        # Score based on relevance and completeness
        score = 0
        if len(answer) > 10:  # Non-trivial response
            score += 1
        if any(word in answer.lower() for word in ["train", "sit", "treat", "owner", "care", "adopt", "new orleans", "music", "food"]):
            score += 1
        if len(answer.split()) > 8:  # Detailed response
            score += 1
            
        instruction_scores.append(score)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Score: {score}/3\n")
    
    # Task 3: Coherence Test
    print("📝 Task 3: Text Coherence")
    coherence_prompts = [
        "Tell me about a typical day with a pet.",
        "Describe what makes a good companion animal.",
        "Explain the importance of routine for pets."
    ]
    
    coherence_scores = []
    for prompt in coherence_prompts:
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        response = generate_response(model, tokenizer, config, full_prompt, max_tokens=100)
        
        if "### Response:\n" in response:
            answer = response.split("### Response:\n")[-1].strip()
        else:
            answer = response.strip()
        
        # Score based on coherence indicators
        score = 0
        if len(answer) > 20:  # Substantial response
            score += 1
        if not any(word in answer.lower() for word in ["<", ">", "###", "unknown", "error"]):  # No formatting artifacts
            score += 1
        if len(set(answer.lower().split())) > 10:  # Vocabulary diversity
            score += 1
            
        coherence_scores.append(score)
        print(f"Prompt: {prompt}")
        print(f"Response: {answer}")
        print(f"Score: {score}/3\n")
    
    # Calculate overall scores
    story_avg = sum(story_scores) / len(story_scores)
    instruction_avg = sum(instruction_scores) / len(instruction_scores)
    coherence_avg = sum(coherence_scores) / len(coherence_scores)
    overall_avg = (story_avg + instruction_avg + coherence_avg) / 3
    
    print(f"📈 SUMMARY for {model_name}")
    print(f"Story Questions: {story_avg:.2f}/3.0")
    print(f"Instruction Following: {instruction_avg:.2f}/3.0")
    print(f"Text Coherence: {coherence_avg:.2f}/3.0")
    print(f"Overall Score: {overall_avg:.2f}/3.0")
    
    return {
        'story_score': story_avg,
        'instruction_score': instruction_avg,
        'coherence_score': coherence_avg,
        'overall_score': overall_avg
    }

def compare_models(model_paths):
    """Compare multiple models."""
    print("🏆 MODEL COMPARISON")
    print("=" * 50)
    
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path.rstrip('/'))
        print(f"\n🔄 Loading {model_name}...")
        
        try:
            model, tokenizer, config = load_model_and_tokenizer(model_path)
            
            # Model info
            param_count = sum(p.numel() for p in model.parameters())
            print(f"📊 Model: {config.n_embd}d, {config.n_head}h, {config.n_layer}l")
            print(f"📊 Parameters: {param_count:,}")
            print(f"📊 Context: {config.block_size}")
            
            # Evaluate
            scores = evaluate_model_on_tasks(model, tokenizer, config, model_name)
            results[model_name] = scores
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            continue
    
    # Final comparison
    print("\n🏆 FINAL COMPARISON")
    print("=" * 50)
    print(f"{'Model':<25} {'Story':<8} {'Instruct':<8} {'Coherence':<10} {'Overall':<8}")
    print("-" * 60)
    
    for model_name, scores in results.items():
        print(f"{model_name:<25} {scores['story_score']:<8.2f} {scores['instruction_score']:<8.2f} "
              f"{scores['coherence_score']:<10.2f} {scores['overall_score']:<8.2f}")
    
    # Recommend best model
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['overall_score'])
        print(f"\n🥇 Best Overall Model: {best_model} (Score: {results[best_model]['overall_score']:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare language models")
    parser.add_argument("models", nargs="+", help="Paths to model directories to compare")
    
    args = parser.parse_args()
    
    # Validate model paths
    valid_paths = []
    for model_path in args.models:
        if os.path.exists(model_path):
            valid_paths.append(model_path)
        else:
            print(f"⚠️  Warning: Model path not found: {model_path}")
    
    if not valid_paths:
        print("❌ No valid model paths provided")
        sys.exit(1)
    
    compare_models(valid_paths)

if __name__ == "__main__":
    main()
