"""
Tokenizer package for BPE (Byte-Pair Encoding) implementation.

This package provides both custom BPE implementation and fast HuggingFace tokenizers.
"""

from .bpe import BPETokenizer

# Try to import fast tokenizer, fallback to custom if not available
try:
    from .fast_tokenizer import FastBPETokenizer
    FAST_TOKENIZER_AVAILABLE = True
except ImportError:
    FAST_TOKENIZER_AVAILABLE = False
    FastBPETokenizer = None

# Default to fast tokenizer if available
if FAST_TOKENIZER_AVAILABLE:
    DefaultTokenizer = FastBPETokenizer
else:
    DefaultTokenizer = BPETokenizer

__all__ = ["BPETokenizer", "DefaultTokenizer"]

if FAST_TOKENIZER_AVAILABLE:
    __all__.append("FastBPETokenizer")
