"""
Data package for dataset handling and preprocessing.

This package provides dataset classes for loading and processing
text data for model training and fine-tuning.
"""

from .dataset import TextDataset
from .instruction_dataset import InstructionDataset, AlpacaDataset, create_instruction_dataset

__all__ = ["TextDataset", "InstructionDataset", "AlpacaDataset", "create_instruction_dataset"]
