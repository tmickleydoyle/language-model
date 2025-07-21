#!/usr/bin/env python3
"""
Unified Data Manager - Fetch data once and serve to both BPE and model training
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from .streaming_data_loader import StreamingDataLoader
from .streaming_dataset import CachedStreamingDataset, StreamingTextDataset


class UnifiedDataManager:
    """Manages data fetching and caching for both tokenizer and model training"""
    
    def __init__(self, 
                 sources: Dict[str, int],
                 cache_dir: str = "data_cache",
                 tokenizer_sample_size: int = 2000,
                 model_cache_size: int = 10000,
                 batch_size: int = 500):
        """
        Args:
            sources: Dict mapping source names to sample counts
            cache_dir: Directory to store cached data
            tokenizer_sample_size: Number of samples for tokenizer training
            model_cache_size: Size of model training cache
            batch_size: Number of samples per cache batch
        """
        self.sources = sources
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.tokenizer_sample_size = tokenizer_sample_size
        self.model_cache_size = model_cache_size
        self.batch_size = batch_size
        
        # Cache files
        self.raw_texts_cache = self.cache_dir / "raw_texts.pkl"
        self.tokenizer_texts_cache = self.cache_dir / "tokenizer_texts.pkl"
        self.metadata_cache = self.cache_dir / "metadata.json"
        self.batch_cache_dir = self.cache_dir / "batches"
        self.batch_cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._raw_texts = None
        self._tokenizer_texts = None
        
    def _write_batch_cache(self, batch_texts: List[str], batch_index: int):
        """Write a batch of texts to incremental cache"""
        batch_file = self.batch_cache_dir / f"batch_{batch_index:04d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_texts, f)
    
    def _load_batch_caches(self) -> List[str]:
        """Load all batch caches and combine them"""
        print("📦 Loading batch caches...")
        all_texts = []
        
        batch_files = sorted(self.batch_cache_dir.glob("batch_*.pkl"))
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch_texts = pickle.load(f)
                    all_texts.extend(batch_texts)
                    print(f"📥 Loaded {len(batch_texts)} texts from {batch_file.name}")
            except Exception as e:
                print(f"⚠️ Error loading {batch_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_texts)} texts from {len(batch_files)} batches")
        return all_texts
    
    def _save_metadata(self):
        """Save metadata about the cached data"""
        batch_files = list(self.batch_cache_dir.glob("batch_*.pkl"))
        
        metadata = {
            "sources": self.sources,
            "tokenizer_sample_size": self.tokenizer_sample_size,
            "model_cache_size": self.model_cache_size,
            "batch_size": self.batch_size,
            "raw_texts_count": len(self._raw_texts) if self._raw_texts else 0,
            "tokenizer_texts_count": len(self._tokenizer_texts) if self._tokenizer_texts else 0,
            "batch_count": len(batch_files)
        }
        
        with open(self.metadata_cache, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata about cached data"""
        if not self.metadata_cache.exists():
            return None
            
        try:
            with open(self.metadata_cache, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _is_cache_valid(self) -> bool:
        """Check if existing cache is valid for current configuration"""
        metadata = self._load_metadata()
        if not metadata:
            return False
        
        # Check if batch caches exist
        batch_files = list(self.batch_cache_dir.glob("batch_*.pkl"))
        has_batch_cache = len(batch_files) > 0
        
        # Check if configuration matches
        config_matches = (metadata.get("sources") == self.sources and
                         metadata.get("tokenizer_sample_size") == self.tokenizer_sample_size)
        
        # Valid if we have either full cache or batch cache with matching config
        return config_matches and (has_batch_cache or 
                                  (self.raw_texts_cache.exists() and self.tokenizer_texts_cache.exists()))
    
    def _fetch_raw_texts(self) -> List[str]:
        """Fetch raw texts from streaming sources with incremental caching"""
        print("🌊 Fetching raw texts from streaming sources...")
        
        loader = StreamingDataLoader()
        texts = []
        batch_counter = 0
        
        # Fetch more than needed to ensure we have enough after filtering
        expanded_sources = {k: int(v * 1.5) for k, v in self.sources.items()}
        target_count = sum(self.sources.values())
        
        try:
            for text in loader.stream_mixed_sources(expanded_sources):
                if len(text.strip()) > 100:  # Filter very short texts
                    texts.append(text)
                    
                # Write incremental cache every batch_size samples
                if len(texts) % self.batch_size == 0:
                    self._write_batch_cache(texts[-self.batch_size:], batch_counter)
                    batch_counter += 1
                    print(f"📥 Fetched {len(texts)} texts... (batch {batch_counter} cached)")
                    
                # Break if we have enough
                if len(texts) >= target_count:
                    # Write final partial batch if needed
                    remaining = len(texts) % self.batch_size
                    if remaining > 0:
                        self._write_batch_cache(texts[-remaining:], batch_counter)
                        batch_counter += 1
                    break
                    
        except Exception as e:
            print(f"⚠️ Error during fetching: {e}")
            # Write partial batch on error
            if texts:
                remaining = len(texts) % self.batch_size
                if remaining > 0:
                    self._write_batch_cache(texts[-remaining:], batch_counter)
                    print(f"💾 Saved {len(texts)} texts in {batch_counter + 1} batches before error")
            if not texts:
                raise
        
        print(f"✅ Fetched {len(texts)} raw texts in {batch_counter + 1} batches")
        return texts
    
    def _prepare_tokenizer_texts(self, raw_texts: List[str]) -> List[str]:
        """Prepare subset of texts for tokenizer training"""
        print(f"📝 Preparing {self.tokenizer_sample_size} texts for tokenizer...")
        
        # Use a diverse sample for tokenizer training
        step = max(1, len(raw_texts) // self.tokenizer_sample_size)
        tokenizer_texts = raw_texts[::step][:self.tokenizer_sample_size]
        
        print(f"✅ Prepared {len(tokenizer_texts)} texts for tokenizer")
        return tokenizer_texts
    
    def ensure_data_ready(self, force_refresh: bool = False) -> bool:
        """
        Ensure data is fetched and cached
        
        Args:
            force_refresh: Force re-fetching even if cache exists
            
        Returns:
            True if data is ready, False if failed
        """
        if not force_refresh and self._is_cache_valid():
            print("✅ Using existing data cache")
            return True
        
        print("🔄 Fetching fresh data...")
        
        try:
            # Fetch raw texts (with incremental caching)
            self._raw_texts = self._fetch_raw_texts()
            
            # Prepare tokenizer texts
            self._tokenizer_texts = self._prepare_tokenizer_texts(self._raw_texts)
            
            # Cache to disk (main cache files)
            print("💾 Caching consolidated data to disk...")
            with open(self.raw_texts_cache, 'wb') as f:
                pickle.dump(self._raw_texts, f)
                
            with open(self.tokenizer_texts_cache, 'wb') as f:
                pickle.dump(self._tokenizer_texts, f)
            
            # Save metadata
            self._save_metadata()
            
            print("✅ Data cached successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to fetch and cache data: {e}")
            # Try to recover from batch caches if available
            if self._try_recover_from_batches():
                print("✅ Recovered data from batch caches")
                return True
            return False
    
    def _try_recover_from_batches(self) -> bool:
        """Try to recover data from batch caches if main cache failed"""
        try:
            batch_files = list(self.batch_cache_dir.glob("batch_*.pkl"))
            if not batch_files:
                return False
            
            print("🔄 Attempting recovery from batch caches...")
            self._raw_texts = self._load_batch_caches()
            
            if self._raw_texts:
                self._tokenizer_texts = self._prepare_tokenizer_texts(self._raw_texts)
                
                # Try to write main cache files
                try:
                    with open(self.raw_texts_cache, 'wb') as f:
                        pickle.dump(self._raw_texts, f)
                    with open(self.tokenizer_texts_cache, 'wb') as f:
                        pickle.dump(self._tokenizer_texts, f)
                    self._save_metadata()
                except Exception as e:
                    print(f"⚠️ Could not write main cache files: {e}")
                
                return True
            return False
            
        except Exception as e:
            print(f"❌ Recovery from batches failed: {e}")
            return False
    
    def get_tokenizer_texts(self) -> List[str]:
        """Get texts for tokenizer training"""
        if self._tokenizer_texts is None:
            if not self.tokenizer_texts_cache.exists():
                # Try to recover from batch caches first
                if not self._try_recover_from_batches():
                    if not self.ensure_data_ready():
                        raise RuntimeError("Failed to prepare tokenizer texts")
            else:
                print("📖 Loading tokenizer texts from cache...")
                with open(self.tokenizer_texts_cache, 'rb') as f:
                    self._tokenizer_texts = pickle.load(f)
        
        return self._tokenizer_texts
    
    def get_model_datasets(self, tokenizer, block_size: int = 128, train_split: float = 0.8):
        """
        Get training datasets for model training
        
        Args:
            tokenizer: Trained tokenizer
            block_size: Context window size
            train_split: Fraction for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self._raw_texts is None:
            if not self.raw_texts_cache.exists():
                # Try to recover from batch caches first
                if not self._try_recover_from_batches():
                    if not self.ensure_data_ready():
                        raise RuntimeError("Failed to prepare model data")
            else:
                print("📖 Loading raw texts from cache...")
                with open(self.raw_texts_cache, 'rb') as f:
                    self._raw_texts = pickle.load(f)
        
        print("🏗️ Creating model datasets from cached data...")
        
        # Create a custom dataset that uses pre-fetched texts
        train_cache_size = int(self.model_cache_size * train_split)
        val_cache_size = self.model_cache_size - train_cache_size
        
        # Split the raw texts to prevent overlap between train and val
        total_texts = len(self._raw_texts)
        train_text_count = int(total_texts * train_split)
        
        train_texts = self._raw_texts[:train_text_count]
        val_texts = self._raw_texts[train_text_count:]
        
        print(f"📊 Split {total_texts} texts: {len(train_texts)} train, {len(val_texts)} val")
        
        # Create datasets with split texts
        train_dataset = PreFetchedStreamingDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            block_size=block_size,
            cache_size=train_cache_size
        )
        
        val_dataset = PreFetchedStreamingDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            block_size=block_size,
            cache_size=val_cache_size
        )
        
        return train_dataset, val_dataset
    
    def clear_cache(self):
        """Clear all cached data"""
        print("🧹 Clearing data cache...")
        
        # Clear main cache files
        for cache_file in [self.raw_texts_cache, self.tokenizer_texts_cache, self.metadata_cache]:
            if cache_file.exists():
                cache_file.unlink()
        
        # Clear batch cache files
        if self.batch_cache_dir.exists():
            for batch_file in self.batch_cache_dir.glob("batch_*.pkl"):
                batch_file.unlink()
        
        self._raw_texts = None
        self._tokenizer_texts = None
        
        print("✅ Cache cleared")


class PreFetchedStreamingDataset(CachedStreamingDataset):
    """Dataset that uses pre-fetched texts instead of streaming"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer,
                 block_size: int = 128,
                 cache_size: int = 10000):
        """
        Args:
            texts: Pre-fetched raw texts
            tokenizer: Tokenizer instance
            block_size: Context window size
            cache_size: Number of samples to cache
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_size = cache_size
        
        # Build cache from pre-fetched texts
        print(f"Building cache of {cache_size} samples from {len(texts)} pre-fetched texts...")
        self._build_cache_from_texts()
    
    def _build_cache_from_texts(self):
        """Build cache from pre-fetched texts"""
        self.cache = []
        current_tokens = []
        
        import random
        random.shuffle(self.texts)  # Randomize order
        
        for text in self.texts:
            if len(self.cache) >= self.cache_size:
                break
                
            try:
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                current_tokens.extend(tokens)
                
                # Create samples from tokens
                while len(current_tokens) >= self.block_size + 1 and len(self.cache) < self.cache_size:
                    sequence = current_tokens[:self.block_size + 1]
                    current_tokens = current_tokens[self.block_size:]
                    
                    # Convert to tensors
                    import torch
                    x = torch.tensor(sequence[:-1], dtype=torch.long)
                    y = torch.tensor(sequence[1:], dtype=torch.long)
                    
                    self.cache.append({'input_ids': x, 'labels': y})
                    
                    if len(self.cache) % 1000 == 0:
                        print(f"Cached {len(self.cache)}/{self.cache_size} samples...")
                        
            except Exception as e:
                print(f"Error processing text: {e}")
                continue
        
        print(f"✅ Cache built with {len(self.cache)} samples")
    
    def refresh_cache(self, refresh_ratio: float = 0.3):
        """Refresh cache with different subset of pre-fetched texts"""
        refresh_count = int(len(self.cache) * refresh_ratio)
        print(f"Refreshing {refresh_count} samples from pre-fetched texts...")
        
        # Create new samples from different texts
        import random
        random.shuffle(self.texts)
        
        new_samples = []
        current_tokens = []
        
        for text in self.texts:
            if len(new_samples) >= refresh_count:
                break
                
            try:
                tokens = self.tokenizer.encode(text)
                current_tokens.extend(tokens)
                
                while len(current_tokens) >= self.block_size + 1 and len(new_samples) < refresh_count:
                    sequence = current_tokens[:self.block_size + 1]
                    current_tokens = current_tokens[self.block_size:]
                    
                    import torch
                    x = torch.tensor(sequence[:-1], dtype=torch.long)
                    y = torch.tensor(sequence[1:], dtype=torch.long)
                    
                    new_samples.append({'input_ids': x, 'labels': y})
                    
            except Exception:
                continue
        
        # Replace random samples in cache
        indices_to_replace = random.sample(range(len(self.cache)), 
                                         min(refresh_count, len(new_samples)))
        
        for i, new_sample in zip(indices_to_replace, new_samples):
            self.cache[i] = new_sample
        
        print(f"✅ Refreshed {len(new_samples)} samples")