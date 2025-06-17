"""Byte Pair Encoding (BPE) tokenizer implementation.

This module provides a minimal byte-level BPE tokenizer that builds vocabulary
by iteratively merging the most frequent character pairs. It handles text
encoding and decoding for use with language models.

Features:
- Byte-level tokenization (handles all Unicode characters)
- Iterative vocabulary building through pair merging
- Configurable vocabulary size and minimum frequency thresholds
- Save/load functionality for trained tokenizers
- Support for both simple and BPE-based encoding schemes

Note: This implementation focuses on simplicity and educational value
while maintaining functionality for language model training.
"""

import logging
import os
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte Pair Encoding tokenizer for text processing.

    This tokenizer starts with byte-level tokens (0-255) and iteratively merges
    the most common pairs to build a vocabulary. It's designed to be simple
    and educational while being functional for language model training.

    Attributes:
        merges: Dictionary mapping token pairs to their merged token ID
        vocab: Dictionary mapping token IDs to their byte representations
        encoder: Dictionary mapping tokens to IDs (for vocab files)
        decoder: Dictionary mapping IDs to tokens (for vocab files)
    """

    def __init__(
        self,
        encoder_file: Optional[str] = None,
        decoder_file: Optional[str] = None,
    ) -> None:
        """Initialize a BPE tokenizer, optionally loading vocab from files.

        Args:
            encoder_file: Path to encoder vocabulary file
            decoder_file: Path to decoder vocabulary file
        """
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}

        if encoder_file and decoder_file:
            self.load_vocab_files(encoder_file, decoder_file)

    def _get_pairs(self, tokens: List[int]) -> CounterType[Tuple[int, int]]:
        """
        Count consecutive token pairs in a sequence.

        Args:
            tokens: List of token IDs

        Returns:
            Counter object with pair frequencies
        """
        pairs: CounterType[Tuple[int, int]] = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_tokens(self, tokens: List[int], pair: Tuple[int, int],
                      new_id: int) -> List[int]:
        """
        Replace all occurrences of a token pair with a new token ID.

        Args:
            tokens: List of token IDs
            pair: Tuple of two token IDs to merge
            new_id: New token ID to replace the pair

        Returns:
            Updated list with merged tokens
        """
        result = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(
        self,
        text: str,
        max_vocab_size: int = 100000,
        verbose: bool = False,
        min_frequency: int = 2,
    ) -> None:
        """Train the BPE model on the given text.

        Args:
            text: Training text corpus
            max_vocab_size: Maximum vocabulary size (including base 256 tokens)
            verbose: Whether to print training progress
            min_frequency: Minimum frequency threshold for merging pairs

        Raises:
            ValueError: If max_vocab_size is less than 256
        """
        if max_vocab_size < 256:
            raise ValueError("max_vocab_size must be at least 256 for byte-level BPE")

        logger.info(f"Training BPE tokenizer on {len(text)} characters")

        # Initialize tokenization
        tokens = self._initialize_tokens(text)
        self._initialize_base_vocab()

        # Perform iterative merging
        self._perform_merging(tokens, max_vocab_size, verbose, min_frequency)

        logger.info(f"Training completed. Final vocabulary size: {len(self.vocab)}")

    def _initialize_tokens(self, text: str) -> List[int]:
        """Initialize tokens from text bytes."""
        text_bytes = text.encode("utf-8")
        return list(text_bytes)

    def _initialize_base_vocab(self) -> None:
        """Initialize vocabulary with all possible bytes."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def _perform_merging(
        self,
        tokens: List[int],
        max_vocab_size: int,
        verbose: bool,
        min_frequency: int,
    ) -> None:
        """Perform iterative merging of most frequent pairs."""
        num_merges = max_vocab_size - 256

        for i in range(num_merges):
            if not self._merge_step(tokens, i, verbose, min_frequency):
                logger.info(
                    f"No more valid pairs to merge. Stopping at {256 + i} tokens."
                )
                break

    def _merge_step(
        self, tokens: List[int], step: int, verbose: bool, min_frequency: int
    ) -> bool:
        """Perform a single merge step.

        Returns:
            True if merge was performed, False if no valid pairs found
        """
        # Count token pairs
        pair_counts = self._get_pairs(tokens)

        # Filter by minimum frequency
        valid_pairs = {
            pair: count
            for pair, count in pair_counts.items()
            if count >= min_frequency
        }

        if not valid_pairs:
            return False

        # Find most frequent pair and merge
        best_pair = max(valid_pairs, key=lambda x: valid_pairs[x])
        new_token_id = 256 + step

        # Update tokens, merges, and vocab
        tokens[:] = self._merge_tokens(tokens, best_pair, new_token_id)
        self.merges[best_pair] = new_token_id
        self.vocab[new_token_id] = (
            self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
        )

        if verbose:
            self._log_merge_info(step, best_pair, new_token_id, valid_pairs)

        return True

    def _log_merge_info(
        self,
        step: int,
        best_pair: Tuple[int, int],
        new_token_id: int,
        valid_pairs: Dict[Tuple[int, int], int],
    ) -> None:
        """Log information about the merge step."""
        pair_str = f"{self.vocab[best_pair[0]]!r} + {self.vocab[best_pair[1]]!r}"
        logger.info(
            f"Merge {step + 1}: {best_pair} -> {new_token_id} "
            f"({pair_str}) frequency: {valid_pairs[best_pair]}"
        )

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs

        Raises:
            RuntimeError: If tokenizer has not been trained or loaded
        """
        # Handle different tokenizer types
        if self.encoder:
            return self._encode_with_vocab_files(text)
        elif self.vocab:
            return self._encode_with_bpe(text)
        else:
            raise RuntimeError(
                "Vocabulary not loaded. Train tokenizer or load vocab files first."
            )

    def _encode_with_vocab_files(self, text: str) -> List[int]:
        """Encode text using vocabulary files (simple word-based)."""
        tokens = []
        words = text.split()  # Simple whitespace tokenization
        for word in words:
            token_id = self._get_token_id_for_word(word)
            tokens.append(token_id)
        return tokens

    def _get_token_id_for_word(self, word: str) -> int:
        """Get token ID for a word, handling unknown words."""
        if word in self.encoder:
            return self.encoder[word]

        # For unknown words, encode as bytes and map to byte-level tokens
        word_bytes = word.encode("utf-8")
        for byte_val in word_bytes:
            if byte_val in self.encoder:
                return self.encoder[byte_val]
            elif str(byte_val) in self.encoder:
                return self.encoder[str(byte_val)]

        # Use the first available token ID if no mapping found
        # Instead of returning raw byte values that may exceed vocab size
        if self.encoder:
            return min(self.encoder.values())  # Return the lowest token ID
        else:
            return 0  # Fallback

    def _encode_with_bpe(self, text: str) -> List[int]:
        """Encode text using BPE merging rules."""
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)

        # Apply merges iteratively
        while len(tokens) >= 2:
            pair_counts = self._get_pairs(tokens)

            # Find the pair with the lowest merge index (highest priority)
            best_pair = min(
                pair_counts.keys(),
                key=lambda pair: self.merges.get(pair, float("inf")),
            )

            # If no valid merge is found, stop
            if best_pair not in self.merges:
                break

            # Apply the merge
            new_token_id = self.merges[best_pair]
            tokens = self._merge_tokens(tokens, best_pair, new_token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string

        Raises:
            ValueError: If tokenizer has not been trained/loaded or
                       unknown token encountered
        """
        if self.decoder:
            return self._decode_with_vocab_files(token_ids)
        elif self.vocab:
            return self._decode_with_bpe(token_ids)
        else:
            raise RuntimeError(
                "Vocabulary not loaded. Train tokenizer or load vocab files first."
            )

    def _decode_with_vocab_files(self, token_ids: List[int]) -> str:
        """Decode token IDs using vocabulary files."""
        try:
            tokens = [self.decoder[token_id] for token_id in token_ids]
            return " ".join(tokens)  # Join words with spaces
        except KeyError as e:
            raise KeyError(f"Unknown token ID: {e}")

    def _decode_with_bpe(self, token_ids: List[int]) -> str:
        """Decode token IDs using BPE vocabulary."""
        try:
            text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
            return text_bytes.decode("utf-8", errors="replace")
        except KeyError as e:
            raise ValueError(f"Unknown token ID: {e}")

    def save(self, file_path: str) -> None:
        """Save the trained tokenizer to files.

        Args:
            file_path: Base path for saving (will create .vocab and .merges files)

        Raises:
            ValueError: If tokenizer hasn't been trained yet
        """
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer must be trained before saving")

        self._save_vocabulary(file_path)
        self._save_merges(file_path)
        logger.info(f"Tokenizer saved to {file_path}.vocab and {file_path}.merges")

    def _save_vocabulary(self, file_path: str) -> None:
        """Save vocabulary to file."""
        vocab_path = f"{file_path}.vocab"
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("# BPE Vocabulary\n")
            for token_id, token_bytes in self.vocab.items():
                f.write(f"{token_id}\t{token_bytes.hex()}\n")

    def _save_merges(self, file_path: str) -> None:
        """Save merge rules to file."""
        merges_path = f"{file_path}.merges"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("# BPE Merges\n")
            for (token1, token2), merged_id in self.merges.items():
                f.write(f"{token1}\t{token2}\t{merged_id}\n")

    def load(self, file_path: str) -> None:
        """Load a trained tokenizer from files.

        Args:
            file_path: Base path for loading (expects .vocab and .merges files)

        Raises:
            FileNotFoundError: If required files don't exist
        """
        vocab_path = f"{file_path}.vocab"
        merges_path = f"{file_path}.merges"

        self._validate_file_paths(vocab_path, merges_path)
        self._load_vocabulary(vocab_path)
        self._load_merges(merges_path)

        logger.info(f"Tokenizer loaded from {vocab_path} and {merges_path}")
        logger.info(f"Vocabulary size: {len(self.vocab)}, Merges: {len(self.merges)}")

    def _validate_file_paths(self, vocab_path: str, merges_path: str) -> None:
        """Validate that required files exist."""
        from ..utils.helpers import _check_file_existence
        _check_file_existence(vocab_path, "Vocabulary file")
        _check_file_existence(merges_path, "Merges file")

    def _load_vocabulary(self, vocab_path: str) -> None:
        """Load vocabulary from file."""
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        token_id, token_hex = parts
                        self.vocab[int(token_id)] = bytes.fromhex(token_hex)

    def _load_merges(self, merges_path: str) -> None:
        """Load merge rules from file."""
        self.merges = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 3:
                        token1, token2, merged_id = parts
                        self.merges[(int(token1), int(token2))] = int(merged_id)

    def load_vocab_files(self, encoder_file: str, decoder_file: str) -> None:
        """Load vocabulary from encoder and decoder files.

        Args:
            encoder_file: Path to the encoder file (token -> id mapping)
            decoder_file: Path to the decoder file (id -> token mapping)

        Raises:
            FileNotFoundError: If vocabulary files don't exist
            ValueError: If files have invalid format
        """
        self._validate_vocab_files(encoder_file, decoder_file)
        self._load_encoder_file(encoder_file)
        self._load_decoder_file(decoder_file)

    def _validate_vocab_files(self, encoder_file: str, decoder_file: str) -> None:
        """Validate that vocabulary files exist."""
        from ..utils.helpers import _check_file_existence
        _check_file_existence(encoder_file, "Encoder file")
        _check_file_existence(decoder_file, "Decoder file")

    def _load_encoder_file(self, encoder_file: str) -> None:
        """Load encoder mapping from file."""
        self.encoder = {}
        try:
            with open(encoder_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line:
                        token, token_id = self._parse_encoder_line(line, encoder_file)
                        self.encoder[token] = token_id
        except UnicodeDecodeError:
            raise ValueError(f"Invalid encoder format in {encoder_file}")

    def _parse_encoder_line(self, line: str, encoder_file: str) -> tuple[str, int]:
        """Parse a single line from encoder file."""
        try:
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                raise ValueError("Invalid format")

            token, token_id = parts
            token = self._unescape_token(token)
            return token, int(token_id)
        except ValueError:
            raise ValueError(f"Invalid encoder format in {encoder_file}: {line}")

    def _load_decoder_file(self, decoder_file: str) -> None:
        """Load decoder mapping from file."""
        self.decoder = {}
        try:
            with open(decoder_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')
                    if line:
                        token_id, token = self._parse_decoder_line(line, decoder_file)
                        self.decoder[token_id] = token
        except UnicodeDecodeError:
            raise ValueError(f"Invalid decoder format in {decoder_file}")

    def _parse_decoder_line(self, line: str, decoder_file: str) -> tuple[int, str]:
        """Parse a single line from decoder file."""
        try:
            parts = line.split(' ', 1)
            if len(parts) != 2:
                raise ValueError("Invalid format")

            token_id, token = parts
            token = self._unescape_token(token)
            return int(token_id), token
        except ValueError:
            raise ValueError(f"Invalid decoder format in {decoder_file}: {line}")

    def _unescape_token(self, token: str) -> str:
        """Unescape special characters in token."""
        return token.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        if self.encoder:
            return len(self.encoder)
        elif self.vocab:
            return len(self.vocab)
        else:
            return 0

    @property
    def is_loaded(self) -> bool:
        """Return True if vocabulary is loaded."""
        return len(self.encoder) > 0 or len(self.vocab) > 0

    def __repr__(self) -> str:
        """Return string representation of the tokenizer."""
        if self.is_loaded:
            return f"<BPETokenizer vocab_size={self.vocab_size}>"
        else:
            return "<BPETokenizer not loaded>"

# For backward compatibility


class BPE(BPETokenizer):
    """Legacy class name for backward compatibility."""

    def __init__(self) -> None:
        """Initialize deprecated BPE class."""
        super().__init__()
        logger.warning(
            "BPE class is deprecated. Please use BPETokenizer instead."
        )
