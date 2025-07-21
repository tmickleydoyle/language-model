"""
Fast Cached Dataset - Optimized for quick loading and training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional
import pickle
from pathlib import Path
import time


class FastCachedDataset(Dataset):
    """Fast dataset that loads pre-tokenized sequences from disk efficiently."""
    
    def __init__(self, 
                 cache_dir: str,
                 tokenizer,
                 block_size: int = 128,
                 max_samples: Optional[int] = None):
        """
        Args:
            cache_dir: Directory containing cached data
            tokenizer: Tokenizer instance
            block_size: Context window size
            max_samples: Maximum samples to load (None = all)
        """
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        print("⚡ Loading fast cache...")
        start_time = time.time()
        
        # Load pre-computed token sequences
        cache_file = self.cache_dir / "token_sequences.npy"
        if cache_file.exists():
            # Use memory-mapped array for fast loading
            self.token_sequences = np.load(cache_file, mmap_mode='r')
            if max_samples:
                self.token_sequences = self.token_sequences[:max_samples]
            print(f"✅ Loaded {len(self.token_sequences)} sequences in {time.time() - start_time:.1f}s")
        else:
            print("❌ No cache found. Building cache...")
            self._build_fast_cache()
            
    def _build_fast_cache(self):
        """Build optimized cache from raw texts."""
        # Load raw texts
        raw_texts_file = self.cache_dir / "raw_texts.pkl"
        if not raw_texts_file.exists():
            raise FileNotFoundError(f"No raw texts found at {raw_texts_file}")
            
        with open(raw_texts_file, 'rb') as f:
            texts = pickle.load(f)
            
        print(f"🔄 Building fast cache from {len(texts)} texts...")
        
        # Tokenize and create sequences efficiently
        all_tokens = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(texts)} texts...")
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            
        # Create training sequences
        sequences = []
        for i in range(0, len(all_tokens) - self.block_size, self.block_size // 2):  # 50% overlap
            seq = all_tokens[i:i + self.block_size + 1]
            if len(seq) == self.block_size + 1:
                sequences.append(seq)
                
        # Convert to numpy array and save
        sequences_array = np.array(sequences, dtype=np.int32)
        cache_file = self.cache_dir / "token_sequences.npy"
        np.save(cache_file, sequences_array)
        
        self.token_sequences = sequences_array
        print(f"✅ Built cache with {len(sequences)} sequences")
        
    def __len__(self):
        return len(self.token_sequences)
        
    def __getitem__(self, idx):
        """Get a training sample."""
        seq = self.token_sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class StreamingFastDataset(Dataset):
    """Even faster dataset that streams from disk without loading everything."""
    
    def __init__(self,
                 cache_dir: str,
                 tokenizer,
                 block_size: int = 128,
                 cache_size: int = 10000):
        """
        Args:
            cache_dir: Directory containing cached batch files
            tokenizer: Tokenizer instance  
            block_size: Context window size
            cache_size: Number of sequences to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_size = cache_size
        
        # Find all batch files
        self.batch_files = sorted(self.cache_dir.glob("batches/batch_*.pkl"))
        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {cache_dir}/batches/")
            
        print(f"⚡ Found {len(self.batch_files)} batch files")
        
        # Load initial cache
        self._load_cache(0)
        
    def _load_cache(self, start_batch: int):
        """Load a subset of batches into memory."""
        print(f"🔄 Loading cache starting from batch {start_batch}...")
        
        self.cache = []
        self.cache_start_idx = 0
        
        # Load batches until we have enough sequences
        for i in range(start_batch, min(start_batch + 10, len(self.batch_files))):
            with open(self.batch_files[i], 'rb') as f:
                texts = pickle.load(f)
                
            # Convert texts to sequences
            for text in texts:
                tokens = self.tokenizer.encode(text)
                # Create sequences with sliding window
                for j in range(0, len(tokens) - self.block_size, self.block_size // 2):
                    seq = tokens[j:j + self.block_size + 1]
                    if len(seq) == self.block_size + 1:
                        self.cache.append(seq)
                        
                if len(self.cache) >= self.cache_size:
                    break
                    
            if len(self.cache) >= self.cache_size:
                break
                
        print(f"✅ Loaded {len(self.cache)} sequences")
        
    def __len__(self):
        # Estimate based on batch files
        return len(self.batch_files) * 1000  # Rough estimate
        
    def __getitem__(self, idx):
        """Get a training sample with dynamic loading."""
        # Simple random sampling from cache
        cache_idx = idx % len(self.cache)
        seq = self.cache[cache_idx]
        
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y