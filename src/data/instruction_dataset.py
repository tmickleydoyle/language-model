"""Instruction fine-tuning dataset for training models on structured Q&A pairs.

This module provides comprehensive dataset classes for handling instruction-following data
including:
- Support for multiple data formats (Alpaca, OpenAI, custom)
- Flexible prompt templating system
- Robust data validation and error handling
- Efficient tokenization and batching
- Smart label masking for instruction vs response tokens
- Memory-efficient data processing pipeline
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any

import torch
from torch.utils.data import Dataset

from ..tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning with structured Q&A pairs.

    Handles loading and formatting of instruction-following datasets in
    various formats including Alpaca, ShareGPT, and custom formats.

    Args:
        data_path: Path to the instruction dataset file (JSON format)
        tokenizer: Tokenizer instance for encoding text
        max_length: Maximum sequence length for training
        instruction_template: Template for formatting instructions
        response_template: Template for formatting responses

    Raises:
        ValueError: If max_length is non-positive or dataset is invalid
        FileNotFoundError: If data_path does not exist
        json.JSONDecodeError: If dataset file contains invalid JSON
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
        self._validate_initialization_params(data_path, tokenizer, max_length)

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.response_template = response_template

        self.data = self._load_and_validate_dataset()
        self.processed_data = self._process_all_examples()

        logger.info(f"Loaded {len(self.data)} instruction examples from {self.data_path}")

    def _validate_initialization_params(
        self,
        data_path: Union[str, Path],
        tokenizer: BPETokenizer,
        max_length: int
    ) -> None:
        """Validate initialization parameters.

        Args:
            data_path: Path to dataset file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length

        Raises:
            ValueError: If parameters are invalid
        """
        if not data_path:
            raise ValueError("data_path cannot be None or empty")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

    def _load_and_validate_dataset(self) -> List[Dict[str, Any]]:
        """Load and validate dataset from JSON file.

        Returns:
            List of validated data examples

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If dataset structure is invalid
        """
        data = self._load_json_data()
        self._validate_dataset_structure(data)
        self._validate_and_clean_examples(data)

        logger.info(f"Successfully validated {len(data)} examples")
        return data

    def _load_json_data(self) -> List[Dict[str, Any]]:
        """Load JSON data from file.

        Returns:
            Raw data loaded from JSON

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded JSON data from {self.data_path}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dataset file {self.data_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read dataset file {self.data_path}: {e}")
            raise RuntimeError(f"Failed to read dataset file {self.data_path}: {e}")

    def _validate_dataset_structure(self, data: Any) -> None:
        """Validate overall dataset structure.

        Args:
            data: Raw data to validate

        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of dictionaries, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Dataset is empty")

    def _validate_and_clean_examples(self, data: List[Dict[str, Any]]) -> None:
        """Validate and clean individual examples.

        Args:
            data: List of examples to validate

        Raises:
            ValueError: If examples are invalid
        """
        required_fields = {'instruction', 'input', 'output'}

        for i, item in enumerate(data):
            self._validate_example_structure(item, i)
            self._validate_required_fields(item, required_fields, i)
            self._clean_example_fields(item, required_fields, i)

    def _validate_example_structure(self, item: Any, index: int) -> None:
        """Validate individual example structure.

        Args:
            item: Example to validate
            index: Example index for error messages

        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(item, dict):
            raise ValueError(f"Example {index} must be a dictionary, got {type(item).__name__}")

    def _validate_required_fields(
        self,
        item: Dict[str, Any],
        required_fields: set,
        index: int
    ) -> None:
        """Validate that example has required fields.

        Args:
            item: Example to validate
            required_fields: Set of required field names
            index: Example index for error messages

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            if len(missing_fields) == 1:
                field = next(iter(missing_fields))
                raise ValueError(f"Missing required field '{field}'")
            else:
                fields_str = ', '.join(sorted(missing_fields))
                raise ValueError(f"Missing required fields: {fields_str}")

    def _clean_example_fields(
        self,
        item: Dict[str, Any],
        required_fields: set,
        index: int
    ) -> None:
        """Clean and normalize example fields.

        Args:
            item: Example to clean (modified in place)
            required_fields: Set of required field names
            index: Example index for logging
        """
        for field in required_fields:
            if not isinstance(item[field], str):
                logger.warning(f"Example {index} field '{field}' is not a string, converting")
                item[field] = str(item[field])

    def _process_all_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Process all raw data examples into tokenized format.

        Returns:
            List of processed examples with tensors

        Raises:
            ValueError: If no valid examples found
        """
        processed = []

        for i, item in enumerate(self.data):
            try:
                processed_item = self._process_single_example(item, i)
                processed.append(processed_item)
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue

        if not processed:
            raise ValueError("No valid examples found in dataset")

        logger.info(f"Successfully processed {len(processed)}/{len(self.data)} examples")
        return processed

    def _process_single_example(
        self,
        item: Dict[str, Any],
        index: int
    ) -> Dict[str, torch.Tensor]:
        """Process a single example into tokenized format.

        Args:
            item: Raw example data
            index: Example index for logging

        Returns:
            Processed example with tensors
        """
        full_text = self._format_example_text(item)
        tokens = self._tokenize_and_truncate(full_text, index)
        return self._create_tensors_from_tokens(tokens, item, full_text)

    def _format_example_text(self, item: Dict[str, Any]) -> str:
        """Format example text using templates.

        Args:
            item: Raw example data

        Returns:
            Formatted text string
        """
        formatted_instruction = self.instruction_template.format(
            instruction=item['instruction'],
            input=item['input']
        )

        formatted_response = self.response_template.format(
            output=item['output']
        )

        return formatted_instruction + formatted_response

    def _tokenize_and_truncate(self, text: str, index: int) -> List[int]:
        """Tokenize text and truncate if necessary.

        Args:
            text: Text to tokenize
            index: Example index for logging

        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.max_length:
            original_length = len(tokens)
            tokens = tokens[:self.max_length]
            logger.warning(
                f"Example {index} truncated from {original_length} to {self.max_length} tokens"
            )

        return tokens

    def _create_tensors_from_tokens(
        self,
        tokens: List[int],
        item: Dict[str, Any],
        full_text: str
    ) -> Dict[str, torch.Tensor]:
        """Create tensor representations from tokens.

        Args:
            tokens: List of token IDs
            item: Original example data
            full_text: Formatted text

        Returns:
            Dictionary with tensor representations
        """
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels_mask = self._create_labels_mask(item, target_ids)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'labels_mask': labels_mask,
            'raw_text': full_text
        }

    def _create_labels_mask(
        self,
        item: Dict[str, Any],
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Create labels mask to train only on response tokens.

        Args:
            item: Original example data
            target_ids: Target token IDs tensor

        Returns:
            Binary mask tensor (1 for response tokens, 0 for instruction tokens)
        """
        formatted_instruction = self.instruction_template.format(
            instruction=item['instruction'],
            input=item['input']
        )

        instruction_tokens = self.tokenizer.encode(formatted_instruction)
        labels_mask = torch.zeros_like(target_ids)

        if len(instruction_tokens) < len(target_ids):
            mask_start = self._find_response_start(
                instruction_tokens, target_ids, item
            )
            labels_mask[mask_start:] = 1

        return labels_mask

    def _find_response_start(
        self,
        instruction_tokens: List[int],
        target_ids: torch.Tensor,
        item: Dict[str, Any]
    ) -> int:
        """Find the start position of response tokens for masking.

        Args:
            instruction_tokens: List of instruction token IDs
            target_ids: Target token IDs tensor
            item: Original example data

        Returns:
            Start index for response tokens
        """
        mask_start = len(instruction_tokens) - 1  # -1 because input_ids is shifted

        # Check for boundary token merging
        if mask_start > 0:
            response_text = self.response_template.format(output=item['output']).strip()
            if response_text:
                mask_start = self._adjust_for_boundary_merging(
                    mask_start, target_ids, response_text
                )

        return max(0, mask_start)

    def _adjust_for_boundary_merging(
        self,
        mask_start: int,
        target_ids: torch.Tensor,
        response_text: str
    ) -> int:
        """Adjust mask start for potential boundary token merging.

        Args:
            mask_start: Initial mask start position
            target_ids: Target token IDs tensor
            response_text: Response text to check

        Returns:
            Adjusted mask start position
        """
        prev_token = target_ids[mask_start - 1].item()

        # Add bounds checking to prevent "Unknown token ID" errors
        vocab_size = self.tokenizer.vocab_size
        if prev_token >= vocab_size:
            logger.warning(f"Token ID {prev_token} exceeds vocabulary size {vocab_size}, skipping boundary adjustment")
            return mask_start

        try:
            prev_decoded = self.tokenizer.decode([prev_token])
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to decode token {prev_token}: {e}, skipping boundary adjustment")
            return mask_start

        response_words = response_text.split()
        if response_words:
            first_word = response_words[0]
            if first_word.lower() in prev_decoded.lower():
                return mask_start - 1

        return mask_start

    def __len__(self) -> int:
        """Return number of examples in dataset.

        Returns:
            Number of processed examples
        """
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example.

        Args:
            idx: Index of example to retrieve

        Returns:
            Dictionary containing tensors for the example

        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.processed_data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.processed_data)}"
            )

        return self.processed_data[idx]

    def get_raw_example(self, idx: int) -> Dict[str, Any]:
        """Get the raw (unprocessed) example at the given index.

        Args:
            idx: Index of example to retrieve

        Returns:
            Raw example dictionary

        Raises:
            IndexError: If index is out of range
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        return self.data[idx]

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader to handle variable-length sequences.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched and padded tensors

        Raises:
            ValueError: If batch is empty
        """
        if not batch:
            raise ValueError("Batch cannot be empty")

        max_len = self._find_max_sequence_length(batch)
        return self._create_padded_batch(batch, max_len)

    def _find_max_sequence_length(self, batch: List[Dict[str, torch.Tensor]]) -> int:
        """Find maximum sequence length in batch.

        Args:
            batch: List of examples

        Returns:
            Maximum sequence length
        """
        return max(len(example['input_ids']) for example in batch)

    def _create_padded_batch(
        self,
        batch: List[Dict[str, torch.Tensor]],
        max_len: int
    ) -> Dict[str, torch.Tensor]:
        """Create padded batch tensors.

        Args:
            batch: List of examples
            max_len: Maximum sequence length for padding

        Returns:
            Dictionary of batched and padded tensors
        """
        batch_size = len(batch)

        # Initialize batch tensors with zeros
        batch_tensors = self._initialize_batch_tensors(batch_size, max_len)

        # Fill batch tensors with example data
        self._fill_batch_tensors(batch, batch_tensors)

        return batch_tensors

    def _initialize_batch_tensors(self, batch_size: int, max_len: int) -> Dict[str, torch.Tensor]:
        """Initialize batch tensors with zeros.

        Args:
            batch_size: Number of examples in batch
            max_len: Maximum sequence length

        Returns:
            Dictionary of initialized tensors
        """
        return {
            'input_ids': torch.zeros(batch_size, max_len, dtype=torch.long),
            'target_ids': torch.zeros(batch_size, max_len, dtype=torch.long),
            'attention_mask': torch.zeros(batch_size, max_len, dtype=torch.long),
            'labels_mask': torch.zeros(batch_size, max_len, dtype=torch.long)
        }

    def _fill_batch_tensors(
        self,
        batch: List[Dict[str, torch.Tensor]],
        batch_tensors: Dict[str, torch.Tensor]
    ) -> None:
        """Fill batch tensors with example data.

        Args:
            batch: List of examples
            batch_tensors: Batch tensors to fill (modified in place)
        """
        for i, example in enumerate(batch):
            seq_len = len(example['input_ids'])
            for key in batch_tensors:
                batch_tensors[key][i, :seq_len] = example[key]


class AlpacaDataset(InstructionDataset):
    """Specialized dataset for Alpaca-format instruction data.

    Uses the standard Alpaca prompt template for consistent formatting
    with the widely-used Alpaca instruction-following dataset format.

    Args:
        data_path: Path to the Alpaca dataset file (JSON format)
        tokenizer: Tokenizer instance for encoding text
        max_length: Maximum sequence length for training
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: BPETokenizer,
        max_length: int = 512
    ) -> None:
        """Initialize with Alpaca-specific formatting using standard template."""
        # Use the default InstructionDataset templates which are Alpaca-compatible
        super().__init__(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        """Format a single example into the Alpaca prompt format.

        Args:
            item: Raw example data

        Returns:
            Formatted prompt string
        """
        formatted_instruction = self.instruction_template.format(
            instruction=item['instruction'],
            input=item['input']
        )
        return formatted_instruction


def create_instruction_dataset(
    data_path: Union[str, Path],
    tokenizer: BPETokenizer,
    max_length: int = 512,
    dataset_format: str = "alpaca"
) -> InstructionDataset:
    """Factory function to create instruction datasets.

    Args:
        data_path: Path to the dataset file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        dataset_format: Format type ('alpaca', 'custom', 'instruction')

    Returns:
        Configured InstructionDataset instance

    Raises:
        ValueError: If dataset_format is not recognized
    """
    fmt = dataset_format.lower()

    if fmt == "alpaca":
        return AlpacaDataset(data_path, tokenizer, max_length)
    elif fmt in ("custom", "instruction"):
        return InstructionDataset(data_path, tokenizer, max_length)
    else:
        supported_formats = ["alpaca", "custom", "instruction"]
        raise ValueError(
            f"Unknown dataset format: {dataset_format}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
