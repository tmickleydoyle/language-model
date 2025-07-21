"""Fast BPE tokenizer using HuggingFace's tokenizers library."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer
    from tokenizers.decoders import BPEDecoder
except ImportError:
    raise ImportError(
        "HuggingFace tokenizers not installed. Install with: pip install tokenizers"
    )

logger = logging.getLogger(__name__)


class FastBPETokenizer:
    """Fast BPE tokenizer using HuggingFace's tokenizers library.
    
    This provides a drop-in replacement for the custom BPE implementation
    with much better performance, especially for large datasets.
    """
    
    def __init__(
        self, 
        tokenizer_path: Optional[Union[str, Path]] = None,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """Initialize the fast BPE tokenizer.
        
        Args:
            tokenizer_path: Path to load existing tokenizer from
            vocab_size: Maximum vocabulary size for training
            min_frequency: Minimum frequency for BPE merges
            special_tokens: List of special tokens to add
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]
        
        if tokenizer_path:
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize a new BPE tokenizer."""
        # Create BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Add normalizers (optional - you can disable these for raw text)
        # self.tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        
        # Don't use pre-tokenizer for byte-level BPE to match custom implementation
        # self.tokenizer.pre_tokenizer = Whitespace()
        
        # Add decoder
        self.tokenizer.decoder = BPEDecoder()
        
        # Add special tokens
        self.tokenizer.add_special_tokens(self.special_tokens)
        
        logger.info("Initialized new BPE tokenizer")
    
    def train(
        self, 
        text: Union[str, List[str]], 
        max_vocab_size: Optional[int] = None,
        verbose: bool = False,
        min_frequency: Optional[int] = None
    ):
        """Train the BPE tokenizer on text data.
        
        Args:
            text: Training text (string or list of strings)
            max_vocab_size: Maximum vocabulary size (uses self.vocab_size if None)
            verbose: Whether to show training progress
            min_frequency: Minimum frequency for merges (uses self.min_frequency if None)
        """
        vocab_size = max_vocab_size or self.vocab_size
        min_freq = min_frequency or self.min_frequency
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=self.special_tokens,
            show_progress=verbose
        )
        
        # Prepare training data
        if isinstance(text, str):
            # Write text to temporary file for training
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                temp_path = f.name
            
            try:
                self.tokenizer.train([temp_path], trainer)
            finally:
                Path(temp_path).unlink()  # Clean up temp file
        else:
            # Train on list of strings
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for line in text:
                    f.write(line + '\n')
                temp_path = f.name
            
            try:
                self.tokenizer.train([temp_path], trainer)
            finally:
                Path(temp_path).unlink()  # Clean up temp file
        
        logger.info(f"Training completed. Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not hasattr(self.tokenizer, 'encode'):
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not hasattr(self.tokenizer, 'decode'):
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.decode(token_ids)
    
    def save(self, file_path: Union[str, Path]):
        """Save the tokenizer to a file.
        
        Args:
            file_path: Path to save the tokenizer
        """
        if not hasattr(self.tokenizer, 'save'):
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        file_path = Path(file_path)
        
        # Save as JSON (HuggingFace format)
        if file_path.suffix == '.json':
            self.tokenizer.save(str(file_path))
        else:
            # Default to .json extension
            json_path = file_path.with_suffix('.json')
            self.tokenizer.save(str(json_path))
        
        logger.info(f"Tokenizer saved to {file_path}")
    
    def load(self, file_path: Union[str, Path]):
        """Load a tokenizer from a file.
        
        Args:
            file_path: Path to load the tokenizer from
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            # Try with .json extension
            if not file_path.suffix:
                file_path = file_path.with_suffix('.json')
        
        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
        
        self.tokenizer = Tokenizer.from_file(str(file_path))
        logger.info(f"Tokenizer loaded from {file_path}")
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        if hasattr(self.tokenizer, 'get_vocab_size'):
            return self.tokenizer.get_vocab_size()
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, size: int):
        """Set the vocabulary size."""
        self._vocab_size = size
    
    @property
    def is_loaded(self) -> bool:
        """Check if tokenizer is loaded/trained."""
        return hasattr(self.tokenizer, 'encode')
    
    def get_vocab(self) -> dict:
        """Get the vocabulary as a dictionary."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.get_vocab()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert ID to token."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.id_to_token(id)
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        if self.is_loaded:
            return f"<FastBPETokenizer vocab_size={self.vocab_size}>"
        else:
            return "<FastBPETokenizer not loaded>"


# Backward compatibility alias
FastBPE = FastBPETokenizer