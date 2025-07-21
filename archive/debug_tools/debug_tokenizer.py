#!/usr/bin/env python3
"""Debug script to check tokenizer training progress."""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tokenizer.bpe import BPETokenizer
from src.data.unified_data_manager import UnifiedDataManager

def debug_tokenizer_training():
    """Debug tokenizer training to see where it stalls."""
    print("🔍 DEBUGGING TOKENIZER TRAINING")
    print("=" * 40)
    
    # Create a small test case first
    print("📝 Testing with small text...")
    small_text = "Hello world! This is a test. " * 100
    
    tokenizer = BPETokenizer()
    start_time = time.time()
    
    print(f"📊 Text length: {len(small_text)} characters")
    print("🏃 Starting tokenizer training...")
    
    try:
        tokenizer.train(small_text, max_vocab_size=1000, verbose=True)
        training_time = time.time() - start_time
        print(f"✅ Small test completed in {training_time:.2f}s")
        print(f"📚 Vocabulary size: {len(tokenizer.vocab)}")
    except Exception as e:
        print(f"❌ Small test failed: {e}")
        return
    
    # Now test with cached data
    print("\n📖 Testing with cached data...")
    
    # Use the same config as the stalled training
    sources = {"wikipedia": 5000}
    
    data_manager = UnifiedDataManager(
        sources=sources,
        cache_dir="wiki_model/data_cache",
        tokenizer_sample_size=500,  # Reduce sample size
        model_cache_size=1000,
        batch_size=500
    )
    
    print("📊 Checking cached data...")
    if not data_manager.ensure_data_ready():
        print("❌ No cached data available")
        return
    
    tokenizer_texts = data_manager.get_tokenizer_texts()
    combined_text = "\n\n".join(tokenizer_texts)
    
    print(f"📊 Cached text stats:")
    print(f"  • Number of texts: {len(tokenizer_texts)}")
    print(f"  • Combined length: {len(combined_text):,} characters")
    print(f"  • Sample: {combined_text[:200]}...")
    
    # Test with progressively larger vocab sizes
    for vocab_size in [1000, 5000, 10000]:
        print(f"\n🧪 Testing vocab_size={vocab_size}...")
        tokenizer = BPETokenizer()
        start_time = time.time()
        
        try:
            # Add timeout mechanism
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Training timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
            tokenizer.train(combined_text, max_vocab_size=vocab_size, verbose=True)
            
            signal.alarm(0)  # Cancel timeout
            
            training_time = time.time() - start_time
            print(f"✅ vocab_size={vocab_size} completed in {training_time:.2f}s")
            print(f"📚 Final vocabulary size: {len(tokenizer.vocab)}")
            
            if training_time > 30:  # If taking too long, stop
                print("⚠️ Training is taking too long, stopping here")
                break
                
        except TimeoutError:
            print(f"⏰ vocab_size={vocab_size} timed out after 60s")
            break
        except Exception as e:
            print(f"❌ vocab_size={vocab_size} failed: {e}")
            break

if __name__ == "__main__":
    debug_tokenizer_training()