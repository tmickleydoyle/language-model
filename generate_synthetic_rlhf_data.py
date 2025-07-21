#!/usr/bin/env python3
"""
Generate Synthetic RLHF Data using Ollama Gemma 3

This script processes OpenWebText files and uses Ollama's gemma3:latest model
to create high-quality synthetic preference pairs for RLHF training.
"""

import json
import pickle
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

class OllamaGenerator:
    """Interface to Ollama for generating synthetic data."""
    
    def __init__(self, model_name: str = "gemma3:latest"):
        self.model_name = model_name
        self._check_ollama_available()
        
    def _check_ollama_available(self):
        """Check if Ollama is available and model exists."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Ollama not found. Please install Ollama first.")
                
            if self.model_name not in result.stdout:
                print(f"⚠️  Model {self.model_name} not found. Pulling...")
                subprocess.run(["ollama", "pull", self.model_name], check=True)
                
            print(f"✅ Ollama model {self.model_name} ready")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error with Ollama: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Ollama."""
        try:
            result = subprocess.run([
                "ollama", "run", self.model_name, prompt
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"⚠️  Generation failed: {result.stderr}")
                return ""
                
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            print("⚠️  Generation timed out")
            return ""
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return ""

class SyntheticRLHFGenerator:
    """Generate synthetic RLHF preference pairs from OpenWebText."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.generator = OllamaGenerator()
        self.output_dir = Path("data/rlhf/datasets/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Templates for different types of synthetic data
        self.prompt_templates = {
            "instruction_following": [
                "Based on this text, create a helpful instruction: '{text_snippet}'",
                "Turn this content into a how-to guide: '{text_snippet}'", 
                "Create a step-by-step explanation from: '{text_snippet}'",
                "Write an educational prompt about: '{text_snippet}'"
            ],
            
            "question_answering": [
                "Generate a question and answer based on: '{text_snippet}'",
                "Create a helpful Q&A from this content: '{text_snippet}'",
                "Write a question that this text answers: '{text_snippet}'",
                "Make an informative question about: '{text_snippet}'"
            ],
            
            "explanation_improvement": [
                "Improve the explanation in: '{text_snippet}'",
                "Make this content clearer: '{text_snippet}'", 
                "Rewrite this to be more helpful: '{text_snippet}'",
                "Simplify this explanation: '{text_snippet}'"
            ],
            
            "creative_expansion": [
                "Expand on this topic creatively: '{text_snippet}'",
                "Add more details to: '{text_snippet}'",
                "Write a follow-up to: '{text_snippet}'",
                "Continue this story/discussion: '{text_snippet}'"
            ]
        }
        
    def load_openwebtext_data(self) -> List[str]:
        """Load OpenWebText data from cache."""
        raw_texts_file = self.cache_dir / "raw_texts.pkl"
        
        if not raw_texts_file.exists():
            # Try to load from batch files
            batch_dir = self.cache_dir / "batches"
            if batch_dir.exists():
                return self._load_from_batches(batch_dir)
            else:
                raise FileNotFoundError(f"No data found in {self.cache_dir}")
                
        print(f"📖 Loading OpenWebText data from {raw_texts_file}")
        with open(raw_texts_file, 'rb') as f:
            texts = pickle.load(f)
            
        print(f"✅ Loaded {len(texts)} OpenWebText documents")
        return texts
    
    def _load_from_batches(self, batch_dir: Path) -> List[str]:
        """Load data from batch files."""
        print(f"📦 Loading from batch files in {batch_dir}")
        texts = []
        
        batch_files = sorted(batch_dir.glob("batch_*.pkl"))
        for batch_file in batch_files[:20]:  # Limit to first 20 batches for manageable size
            try:
                with open(batch_file, 'rb') as f:
                    batch_texts = pickle.load(f)
                    texts.extend(batch_texts)
                    print(f"📥 Loaded {len(batch_texts)} texts from {batch_file.name}")
            except Exception as e:
                print(f"⚠️  Error loading {batch_file}: {e}")
                
        print(f"✅ Loaded {len(texts)} total texts from batches")
        return texts
    
    def extract_text_snippet(self, text: str, max_length: int = 300) -> str:
        """Extract a meaningful snippet from text."""
        # Clean up text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try to find a complete sentence or paragraph
        sentences = text.split('.')
        snippet = ""
        
        for sentence in sentences:
            if len(snippet + sentence) < max_length:
                snippet += sentence + "."
            else:
                break
                
        if len(snippet) < 50:  # Too short, take first max_length chars
            snippet = text[:max_length]
            
        return snippet.strip()
    
    def generate_preference_pair(self, text: str, category: str) -> Optional[Dict[str, Any]]:
        """Generate a preference pair from OpenWebText content."""
        
        # Extract snippet
        snippet = self.extract_text_snippet(text)
        if len(snippet) < 50:
            return None
            
        # Choose template
        template = random.choice(self.prompt_templates[category])
        base_prompt = template.format(text_snippet=snippet)
        
        print(f"🎯 Generating {category} pair...")
        
        # Generate high-quality response
        quality_prompt = f"""Please provide a helpful, detailed, and well-structured response to: {base_prompt}

Make sure your response is:
- Clear and informative
- Well-organized with proper structure
- Helpful to the reader
- Accurate and factual"""

        chosen_response = self.generator.generate(quality_prompt)
        
        if not chosen_response:
            return None
            
        # Generate lower-quality response
        poor_prompts = [
            f"Give a brief, vague response to: {base_prompt}",
            f"Answer this poorly with minimal effort: {base_prompt}",
            f"Provide an unclear, unhelpful response to: {base_prompt}",
            f"Give a confusing answer to: {base_prompt}"
        ]
        
        poor_prompt = random.choice(poor_prompts)
        rejected_response = self.generator.generate(poor_prompt)
        
        if not rejected_response:
            # Create a simple poor response
            rejected_response = "I don't know much about this topic. It's complicated."
            
        # Create preference example
        return {
            "prompt": base_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "metadata": {
                "source": "ollama_synthetic",
                "category": category,
                "model": self.generator.model_name,
                "quality": "helpful_vs_vague",
                "chosen_length": len(chosen_response.split()),
                "rejected_length": len(rejected_response.split()),
                "source_snippet": snippet[:100] + "..." if len(snippet) > 100 else snippet
            }
        }
    
    def generate_dataset(self, num_files: int = 100, examples_per_file: int = 3):
        """Generate synthetic RLHF dataset."""
        
        print("🤖 SYNTHETIC RLHF DATA GENERATION")
        print("=" * 60)
        print(f"Target: {num_files} files × {examples_per_file} examples = {num_files * examples_per_file} total")
        
        # Load OpenWebText data
        texts = self.load_openwebtext_data()
        
        if len(texts) < num_files:
            print(f"⚠️  Only {len(texts)} texts available, using all")
            num_files = len(texts)
            
        # Shuffle for variety
        random.shuffle(texts)
        
        # Generate examples
        all_examples = []
        categories = list(self.prompt_templates.keys())
        
        for i, text in enumerate(texts[:num_files]):
            print(f"\n📄 Processing file {i+1}/{num_files}")
            
            for j in range(examples_per_file):
                category = categories[j % len(categories)]
                
                try:
                    example = self.generate_preference_pair(text, category)
                    if example:
                        all_examples.append(example)
                        print(f"  ✅ Generated {category} example ({len(all_examples)} total)")
                    else:
                        print(f"  ⚠️  Failed to generate {category} example")
                        
                    # Small delay to be nice to Ollama
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  ❌ Error generating example: {e}")
                    continue
                    
            # Save progress periodically
            if (i + 1) % 10 == 0:
                self._save_progress(all_examples, f"progress_{i+1}")
                
        # Save final dataset
        timestamp = int(time.time())
        output_file = self.output_dir / f"ollama_synthetic_preferences_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(all_examples, f, indent=2)
            
        print(f"\n✅ Generated {len(all_examples)} synthetic examples")
        print(f"💾 Saved to: {output_file}")
        
        # Create summary
        self._create_summary(all_examples, output_file)
        
        return str(output_file)
    
    def _save_progress(self, examples: List[Dict], suffix: str):
        """Save progress to avoid losing work."""
        progress_file = self.output_dir / f"synthetic_progress_{suffix}.json"
        with open(progress_file, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"💾 Progress saved: {len(examples)} examples")
    
    def _create_summary(self, examples: List[Dict], output_file: Path):
        """Create a summary of the generated dataset."""
        
        # Analyze categories
        categories = {}
        quality_types = {}
        total_chosen_length = 0
        total_rejected_length = 0
        
        for example in examples:
            metadata = example.get("metadata", {})
            
            category = metadata.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            
            quality_type = metadata.get("quality", "unknown") 
            quality_types[quality_type] = quality_types.get(quality_type, 0) + 1
            
            total_chosen_length += metadata.get("chosen_length", 0)
            total_rejected_length += metadata.get("rejected_length", 0)
            
        summary = {
            "total_examples": len(examples),
            "categories": categories,
            "quality_types": quality_types,
            "avg_chosen_length": total_chosen_length / len(examples) if examples else 0,
            "avg_rejected_length": total_rejected_length / len(examples) if examples else 0,
            "generation_model": "gemma3:latest",
            "source": "openwebtext_synthetic",
            "file": str(output_file.name)
        }
        
        summary_file = self.output_dir / f"summary_{output_file.stem}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📊 DATASET SUMMARY:")
        print(f"Total examples: {summary['total_examples']}")
        print(f"Categories: {summary['categories']}")
        print(f"Avg chosen length: {summary['avg_chosen_length']:.1f} words")
        print(f"Avg rejected length: {summary['avg_rejected_length']:.1f} words")
        print(f"Summary saved: {summary_file}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic RLHF data using Ollama")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="openwebtext_only/data_cache",
        help="OpenWebText cache directory"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=100,
        help="Number of OpenWebText files to process"
    )
    parser.add_argument(
        "--examples-per-file",
        type=int, 
        default=3,
        help="Number of examples to generate per file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:latest",
        help="Ollama model to use"
    )
    
    args = parser.parse_args()
    
    # Check cache directory exists
    if not Path(args.cache_dir).exists():
        print(f"❌ Cache directory not found: {args.cache_dir}")
        print("Available cache directories:")
        for cache_dir in Path(".").glob("*/data_cache"):
            print(f"  - {cache_dir}")
        return
        
    # Create generator
    generator = SyntheticRLHFGenerator(args.cache_dir)
    generator.generator.model_name = args.model
    
    # Generate dataset
    output_file = generator.generate_dataset(
        num_files=args.num_files,
        examples_per_file=args.examples_per_file
    )
    
    print(f"\n🎉 Synthetic RLHF dataset generation complete!")
    print(f"📁 Output: {output_file}")
    print(f"\n💡 Usage:")
    print(f"python scripts/train_rlhf_organized.py --dataset {output_file} --method dpo")

if __name__ == "__main__":
    main()