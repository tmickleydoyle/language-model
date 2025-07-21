#!/usr/bin/env python3
"""
Streaming Dataset - PyTorch dataset that works with streaming data sources
"""

import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Iterator, Optional, Dict, Any, List
import random
from .streaming_data_loader import create_streaming_text_generator, StreamingDataLoader

class StreamingTextDataset(IterableDataset):
    """PyTorch IterableDataset for streaming text data without local storage"""
    
    def __init__(self, 
                 sources: Dict[str, int],
                 tokenizer,
                 block_size: int = 128,
                 chunk_size: int = 50000,
                 max_samples: Optional[int] = None):
        """
        Args:
            sources: Dict mapping source names to sample counts
            tokenizer: Tokenizer instance
            block_size: Context window size
            chunk_size: Size of text chunks to process at once
            max_samples: Maximum number of samples to generate
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through streaming data and yield tokenized sequences"""
        
        text_generator = create_streaming_text_generator(
            self.sources, 
            chunk_size=self.chunk_size
        )
        
        sample_count = 0
        current_tokens = []
        
        for text_chunk in text_generator:
            if self.max_samples and sample_count >= self.max_samples:
                break
                
            # Tokenize the chunk
            try:
                chunk_tokens = self.tokenizer.encode(text_chunk)
                current_tokens.extend(chunk_tokens)
                
                # Yield complete sequences
                while len(current_tokens) >= self.block_size + 1:
                    if self.max_samples and sample_count >= self.max_samples:
                        break
                        
                    # Extract sequence
                    sequence = current_tokens[:self.block_size + 1]
                    current_tokens = current_tokens[self.block_size:]
                    
                    # Convert to tensors
                    x = torch.tensor(sequence[:-1], dtype=torch.long)
                    y = torch.tensor(sequence[1:], dtype=torch.long)
                    
                    yield {'input_ids': x, 'labels': y}
                    sample_count += 1
                    
            except Exception as e:
                print(f"Error processing text chunk: {e}")
                continue

class CachedStreamingDataset(Dataset):
    """Dataset that caches streaming data for multiple epochs"""
    
    def __init__(self,
                 sources: Dict[str, int],
                 tokenizer,
                 block_size: int = 128,
                 cache_size: int = 10000,
                 chunk_size: int = 50000):
        """
        Args:
            sources: Dict mapping source names to sample counts  
            tokenizer: Tokenizer instance
            block_size: Context window size
            cache_size: Number of samples to cache
            chunk_size: Size of text chunks to process at once
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        
        # Pre-populate cache
        print(f"Building cache of {cache_size} samples...")
        self._build_cache()
        
    def _build_cache(self):
        """Build initial cache from streaming data"""
        self.cache = []
        
        streaming_dataset = StreamingTextDataset(
            sources=self.sources,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            chunk_size=self.chunk_size,
            max_samples=self.cache_size
        )
        
        for i, sample in enumerate(streaming_dataset):
            self.cache.append(sample)
            if (i + 1) % 1000 == 0:
                print(f"Cached {i + 1}/{self.cache_size} samples...")
        
        print(f"✅ Cache built with {len(self.cache)} samples")
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        return self.cache[idx]
    
    def refresh_cache(self, refresh_ratio: float = 0.3):
        """Refresh a portion of the cache with new streaming data"""
        refresh_count = int(len(self.cache) * refresh_ratio)
        
        print(f"Refreshing {refresh_count} samples in cache...")
        
        # Get new samples
        streaming_dataset = StreamingTextDataset(
            sources=self.sources,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            chunk_size=self.chunk_size,
            max_samples=refresh_count
        )
        
        new_samples = []
        for sample in streaming_dataset:
            new_samples.append(sample)
            if len(new_samples) >= refresh_count:
                break
        
        # Randomly replace samples in cache
        indices_to_replace = random.sample(range(len(self.cache)), 
                                         min(refresh_count, len(new_samples)))
        
        for i, new_sample in zip(indices_to_replace, new_samples):
            self.cache[i] = new_sample
        
        print(f"✅ Refreshed {len(new_samples)} samples")
    
    def get_batch(
        self, batch_size: int, device: str = "cpu"
    ) -> tuple:
        """Generate a batch of data for training or validation.

        Args:
            batch_size: Number of samples in the batch
            device: Device to place tensors on

        Returns:
            Tuple of (input_tensor, target_tensor) with shape (batch_size, block_size)

        Raises:
            ValueError: If cache is empty
        """
        if len(self.cache) == 0:
            raise ValueError("Cache is empty - cannot generate batches")

        # Generate random indices for batch
        indices = torch.randint(0, len(self.cache), (batch_size,))
        
        # Get samples and extract input_ids and labels
        x_list = []
        y_list = []
        
        for idx in indices:
            sample = self.cache[idx]
            x_list.append(sample['input_ids'])
            y_list.append(sample['labels'])
        
        # Stack into batch tensors
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        # Move to device and return
        return x.to(device), y.to(device)

def create_streaming_datasets(sources: Dict[str, int],
                            tokenizer,
                            block_size: int = 128,
                            train_split: float = 0.8,
                            use_cache: bool = True,
                            cache_size: int = 10000) -> tuple:
    """
    Create train/validation datasets from streaming sources
    
    Args:
        sources: Dict mapping source names to sample counts
        tokenizer: Tokenizer instance  
        block_size: Context window size
        train_split: Fraction of data for training
        use_cache: Whether to use cached dataset
        cache_size: Size of cache if using cached dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    
    if use_cache:
        # Split cache between train and val
        train_cache_size = int(cache_size * train_split)
        val_cache_size = cache_size - train_cache_size
        
        train_dataset = CachedStreamingDataset(
            sources=sources,
            tokenizer=tokenizer,
            block_size=block_size,
            cache_size=train_cache_size
        )
        
        val_dataset = CachedStreamingDataset(
            sources=sources,
            tokenizer=tokenizer,
            block_size=block_size,
            cache_size=val_cache_size
        )
        
    else:
        # Use streaming datasets directly
        train_samples = sum(sources.values()) * train_split
        val_samples = sum(sources.values()) * (1 - train_split)
        
        train_dataset = StreamingTextDataset(
            sources=sources,
            tokenizer=tokenizer,
            block_size=block_size,
            max_samples=int(train_samples)
        )
        
        val_dataset = StreamingTextDataset(
            sources=sources,
            tokenizer=tokenizer,
            block_size=block_size,
            max_samples=int(val_samples)
        )
    
    return train_dataset, val_dataset

# Example configurations for different data source mixes
DATASET_CONFIGS = {
    "small_mixed": {
        "wikipedia": 500,
        "openwebtext": 500
    },
    
    "medium_mixed": {
        "wikipedia": 5000,
        "openwebtext": 10000,
        "pile": 5000
    },
    
    "large_mixed": {
        "wikipedia": 10000,
        "openwebtext": 30000,
        "pile": 20000,
        "c4": 40000
    },
    
    "wikipedia_only": {
        "wikipedia": 500
    },
    
    "web_focused": {
        "openwebtext": 25000,
        "c4": 25000
    },

    "openwebtext_only": {
        "openwebtext": 50000
    },
}