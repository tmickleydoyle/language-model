"""
Instruction fine-tuning dataset for training models on structured Q&A pairs.

This module provides dataset classes for handling instruction-following data
in various formats (Alpaca, OpenAI, etc.) for supervised fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from torch.utils.data import Dataset

from ..tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning with structured Q&A pairs.
    
    Handles loading and formatting of instruction-following datasets in
    various formats including Alpaca, ShareGPT, and custom formats.
    
    Args:
        data_path: Path to the instruction dataset file (JSON format)
        tokenizer: Tokenizer instance for encoding text
        max_length: Maximum sequence length for training
        instruction_template: Template for formatting instructions
        response_template: Template for formatting responses
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: BPETokenizer,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        response_template: str = "{output}"
    ) -> None:
        """Initialize the instruction dataset."""
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        
        # Validate max_length
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        
        self.instruction_template = instruction_template
        self.response_template = response_template
        
        # Load and process the dataset
        self.data = self._load_dataset()
        self.processed_data = self._process_dataset()
        
        logger.info(f"Loaded {len(self.data)} instruction examples from {self.data_path}")
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file with validation."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded JSON data from {self.data_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file {self.data_path}: {e}")
            raise ValueError(f"Invalid JSON in dataset file {self.data_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to read dataset file {self.data_path}: {e}")
            raise RuntimeError(f"Failed to read dataset file {self.data_path}: {e}")
        
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of dictionaries, got {type(data).__name__}")
        
        if len(data) == 0:
            raise ValueError("Dataset is empty")
        
        # Validate required fields
        required_fields = {'instruction', 'input', 'output'}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Example {i} must be a dictionary, got {type(item).__name__}")
            
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(f"Example {i} missing required field(s): {', '.join(missing_fields)}")
            
            # Validate field types
            for field in required_fields:
                if not isinstance(item[field], str):
                    logger.warning(f"Example {i} field '{field}' is not a string, converting to string")
                    item[field] = str(item[field])
        
        logger.info(f"Successfully validated {len(data)} examples")
        return data
    
    def _process_dataset(self) -> List[Dict[str, torch.Tensor]]:
        """Process raw data into tokenized format."""
        processed = []
        
        for i, item in enumerate(self.data):
            try:
                # Format the instruction
                formatted_instruction = self.instruction_template.format(
                    instruction=item['instruction'],
                    input=item['input']
                )
                
                # Format the response
                formatted_response = self.response_template.format(
                    output=item['output']
                )
                
                # Combine instruction and response
                full_text = formatted_instruction + formatted_response
                
                # Tokenize
                tokens = self.tokenizer.encode(full_text)
                
                # Truncate if necessary
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                    logger.warning(f"Example {i} truncated from {len(tokens)} to {self.max_length} tokens")
                
                # Create input and target sequences
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                target_ids = torch.tensor(tokens[1:], dtype=torch.long)
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = torch.ones_like(input_ids)
                
                # Create labels mask (only train on response tokens, not instruction)
                instruction_tokens = self.tokenizer.encode(formatted_instruction)
                labels_mask = torch.zeros_like(target_ids)
                if len(instruction_tokens) < len(target_ids):
                    labels_mask[len(instruction_tokens)-1:] = 1
                
                processed.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'attention_mask': attention_mask,
                    'labels_mask': labels_mask,
                    'raw_text': full_text
                })
                
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        if not processed:
            raise ValueError("No valid examples found in dataset")
        
        logger.info(f"Successfully processed {len(processed)}/{len(self.data)} examples")
        return processed
    
    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        if idx >= len(self.processed_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.processed_data)}")
        
        return self.processed_data[idx]
    
    def get_raw_example(self, idx: int) -> Dict[str, Any]:
        """Get the raw (unprocessed) example at the given index."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        return self.data[idx]
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader to handle variable-length sequences.
        
        Args:
            batch: List of examples from __getitem__
            
        Returns:
            Batched and padded tensors
        """
        # Find max length in batch
        max_len = max(len(example['input_ids']) for example in batch)
        
        # Initialize batch tensors
        batch_size = len(batch)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        target_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        # Fill batch tensors
        for i, example in enumerate(batch):
            seq_len = len(example['input_ids'])
            input_ids[i, :seq_len] = example['input_ids']
            target_ids[i, :seq_len] = example['target_ids']
            attention_mask[i, :seq_len] = example['attention_mask']
            labels_mask[i, :seq_len] = example['labels_mask']
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'labels_mask': labels_mask
        }


class AlpacaDataset(InstructionDataset):
    """
    Specialized dataset for Alpaca-format instruction data.
    
    Uses the standard Alpaca prompt template for consistent formatting.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: BPETokenizer,
        max_length: int = 512
    ) -> None:
        """Initialize with Alpaca-specific formatting."""
        alpaca_instruction_template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        )
        
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            instruction_template=alpaca_instruction_template,
            response_template="{output}"
        )


def create_instruction_dataset(
    data_path: Union[str, Path],
    tokenizer: BPETokenizer,
    max_length: int = 512,
    dataset_format: str = "alpaca"
) -> InstructionDataset:
    """
    Factory function to create instruction datasets.
    
    Args:
        data_path: Path to the dataset file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        dataset_format: Format type ('alpaca', 'custom')
        
    Returns:
        Configured InstructionDataset instance
    """
    if dataset_format.lower() == "alpaca":
        return AlpacaDataset(data_path, tokenizer, max_length)
    elif dataset_format.lower() == "custom":
        return InstructionDataset(data_path, tokenizer, max_length)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
