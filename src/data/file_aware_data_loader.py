#!/usr/bin/env python3
"""
File-aware data loading that respects document boundaries.
"""

import os
import random
from typing import List, Tuple
from ..tokenizer import BPETokenizer
from .dataset import TextDataset
import torch

def load_files_separately(data_path: str) -> List[Tuple[str, str]]:
    """Load all text files as separate documents.
    
    Returns:
        List of (filename, content) tuples
    """
    files_content = []
    
    if os.path.isfile(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        files_content.append((os.path.basename(data_path), content))
    elif os.path.isdir(data_path):
        txt_files = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')])
        
        for txt_file in txt_files:
            file_path = os.path.join(data_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            files_content.append((txt_file, content))
    
    return files_content

def create_file_aware_split(files_content: List[Tuple[str, str]], 
                           train_split: float = 0.8) -> Tuple[str, str]:
    """Create train/val split that respects file boundaries.
    
    Args:
        files_content: List of (filename, content) tuples
        train_split: Fraction of files to use for training
        
    Returns:
        (train_text, val_text) tuple
    """
    # Shuffle files for random split
    files_shuffled = files_content.copy()
    random.shuffle(files_shuffled)
    
    # Split by number of files, not characters
    split_idx = int(len(files_shuffled) * train_split)
    
    train_files = files_shuffled[:split_idx]
    val_files = files_shuffled[split_idx:]
    
    # Concatenate within each split
    train_text = ""
    for filename, content in train_files:
        train_text += content + "\n\n"  # Add file separator
        
    val_text = ""
    for filename, content in val_files:
        val_text += content + "\n\n"  # Add file separator
    
    print(f"📊 File-aware split:")
    print(f"   Train: {len(train_files)} files, {len(train_text):,} characters")
    print(f"   Val: {len(val_files)} files, {len(val_text):,} characters")
    
    return train_text, val_text

class FileAwareTextDataset(TextDataset):
    """TextDataset that can optionally respect file boundaries in sequences."""
    
    def __init__(self, files_content: List[Tuple[str, str]], tokenizer, block_size: int, 
                 respect_boundaries: bool = True):
        """
        Args:
            files_content: List of (filename, content) tuples
            tokenizer: BPE tokenizer
            block_size: Context window size
            respect_boundaries: If True, sequences won't cross file boundaries
        """
        self.files_content = files_content
        self.respect_boundaries = respect_boundaries
        self.file_boundaries = []  # Track where each file starts/ends in token sequence
        
        # Concatenate all content with boundary markers
        full_text = ""
        char_pos = 0
        
        for filename, content in files_content:
            file_start = char_pos
            full_text += content + "\n\n"
            char_pos = len(full_text)
            self.file_boundaries.append((file_start, char_pos - 2))  # Exclude separator
        
        # Initialize parent class
        super().__init__(text=full_text, tokenizer=tokenizer, block_size=block_size)
        
        if respect_boundaries:
            self._compute_valid_sequences()
    
    def _compute_valid_sequences(self):
        """Compute valid sequence start positions that don't cross file boundaries."""
        self.valid_starts = []
        
        # Convert character boundaries to token boundaries (approximate)
        for file_start_char, file_end_char in self.file_boundaries:
            # Find approximate token positions for this file
            # This is approximate since tokenization can change character->token mapping
            file_start_token = int(file_start_char * len(self.tokens) / len(self.text))
            file_end_token = int(file_end_char * len(self.tokens) / len(self.text))
            
            # Add valid sequence starts within this file
            for start_pos in range(file_start_token, 
                                 max(file_start_token, file_end_token - self.block_size)):
                if start_pos + self.block_size < file_end_token:
                    self.valid_starts.append(start_pos)
        
        print(f"📐 File boundary awareness: {len(self.valid_starts)} valid sequences "
              f"(vs {len(self.tokens) - self.block_size} without boundaries)")
    
    def __len__(self) -> int:
        """Return number of valid sequences."""
        if self.respect_boundaries and hasattr(self, 'valid_starts'):
            return len(self.valid_starts)
        else:
            return super().__len__()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample, respecting file boundaries if enabled."""
        if self.respect_boundaries and hasattr(self, 'valid_starts'):
            if idx >= len(self.valid_starts):
                raise IndexError(f"Index {idx} out of range")
            
            actual_start = self.valid_starts[idx]
            x = self.tokens[actual_start:actual_start + self.block_size].clone()
            y = self.tokens[actual_start + 1:actual_start + self.block_size + 1].clone()
            return x, y
        else:
            return super().__getitem__(idx)

# Example usage function
def create_file_aware_datasets(data_path: str, tokenizer: BPETokenizer, 
                              block_size: int, train_split: float = 0.8,
                              respect_boundaries: bool = True):
    """Create train/val datasets that respect file boundaries.
    
    Args:
        data_path: Path to data file or directory
        tokenizer: Trained tokenizer
        block_size: Context window size
        train_split: Fraction of files for training
        respect_boundaries: Whether to prevent sequences from crossing files
        
    Returns:
        (train_dataset, val_dataset) tuple
    """
    # Load files separately
    files_content = load_files_separately(data_path)
    
    # Create file-aware split
    train_text, val_text = create_file_aware_split(files_content, train_split)
    
    if respect_boundaries:
        # Create datasets that respect file boundaries
        train_files = files_content[:int(len(files_content) * train_split)]
        val_files = files_content[int(len(files_content) * train_split):]
        
        train_dataset = FileAwareTextDataset(train_files, tokenizer, block_size, True)
        val_dataset = FileAwareTextDataset(val_files, tokenizer, block_size, True)
    else:
        # Use standard datasets
        train_dataset = TextDataset(text=train_text, tokenizer=tokenizer, block_size=block_size)
        val_dataset = TextDataset(text=val_text, tokenizer=tokenizer, block_size=block_size)
    
    return train_dataset, val_dataset
