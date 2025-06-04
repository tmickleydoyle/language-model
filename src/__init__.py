"""GPT Language Model Implementation.

A PyTorch implementation of a GPT (Generative Pre-trained Transformer) model
with Byte-Pair Encoding (BPE) tokenization.
"""

__version__ = "1.0.0"
__author__ = "Language Model Team"

# Make subpackages easily accessible
from .tokenizer import BPETokenizer
from .model import GPTLanguageModel
from .training import Trainer
from .data import TextDataset
from .utils import count_parameters
from .config import Config, setup_logging

__all__ = [
    "Config",
    "GPTLanguageModel",
    "BPETokenizer",
    "TextDataset",
    "Trainer",
    "setup_logging",
    "count_parameters",
]
