"""
Byte Pair Encoding (BPE) tokenizer implementation.

This module provides a minimal byte-level BPE tokenizer that builds vocabulary
by iteratively merging the most frequent character pairs. It handles text
encoding and decoding for use with language models.

Note: This implementation does not handle regex splitting patterns or special tokens.
"""

import logging
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional, Counter as CounterType

logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer for text processing.

    This tokenizer starts with byte-level tokens (0-255) and iteratively merges
    the most common pairs to build a vocabulary. It's designed to be simple
    and educational while being functional for language model training.

    Attributes:
        merges: Dictionary mapping token pairs to their merged token ID
        vocab: Dictionary mapping token IDs to their byte representations
    """

    def __init__(self, encoder_file: Optional[str] = None,
                 decoder_file: Optional[str] = None) -> None:
        """Initialize a BPE tokenizer, optionally loading vocab from files."""
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

    def train(self, text: str, max_vocab_size: int = 1000,
              verbose: bool = False, min_frequency: int = 2) -> None:
        """
        Train the BPE model on the given text.

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

        # Initialize with byte-level tokens
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)

        # Initialize vocabulary with all possible bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        num_merges = max_vocab_size - 256

        for i in range(num_merges):
            # Count token pairs
            pair_counts = self._get_pairs(tokens)

            # Filter by minimum frequency
            valid_pairs = {
                pair: count for pair, count in pair_counts.items()
                if count >= min_frequency
            }

            if not valid_pairs:
                logger.info(
                    f"No more valid pairs to merge. Stopping at {256 + i} tokens.")
                break

            # Find most frequent pair
            best_pair = max(valid_pairs, key=lambda x: valid_pairs[x])
            new_token_id = 256 + i

            # Merge tokens
            tokens = self._merge_tokens(tokens, best_pair, new_token_id)

            # Update vocabulary and merges
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

            if verbose:
                pair_str = f"{self.vocab[best_pair[0]]!r} + {self.vocab[best_pair[1]]!r}"
                logger.info(
                    f"Merge {i + 1}: {best_pair} -> {new_token_id} "
                    f"({pair_str}) frequency: {valid_pairs[best_pair]}"
                )

        logger.info(f"Training completed. Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs

        Raises:
            ValueError: If tokenizer has not been trained or loaded
        """
        # If using vocab files (encoder/decoder), use simple lookup
        if self.encoder:
            tokens = []
            words = text.split()  # Simple whitespace tokenization
            for word in words:
                if word in self.encoder:
                    tokens.append(self.encoder[word])
                # For unknown words, you might want to handle differently
                # For now, we'll just skip them or use a default token
            return tokens

        # Original BPE encoding logic for trained tokenizer
        if not self.vocab:
            raise RuntimeError(
                "Vocabulary not loaded. Train tokenizer or load vocab files first.")

        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)

        # Apply merges iteratively
        while len(tokens) >= 2:
            pair_counts = self._get_pairs(tokens)

            # Find the pair with the lowest merge index (highest priority)
            best_pair = min(
                pair_counts.keys(),
                key=lambda pair: self.merges.get(pair, float('inf'))
            )

            # If no valid merge is found, stop
            if best_pair not in self.merges:
                break

            # Apply the merge
            new_token_id = self.merges[best_pair]
            tokens = self._merge_tokens(tokens, best_pair, new_token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text string

        Raises:
            ValueError: If tokenizer has not been trained/loaded or
                       unknown token encountered
        """
        # If using vocab files (encoder/decoder), use simple lookup
        if self.decoder:
            try:
                tokens = [self.decoder[token_id] for token_id in token_ids]
                return " ".join(tokens)  # Join words with spaces
            except KeyError as e:
                raise KeyError(f"Unknown token ID: {e}")

        # Original BPE decoding logic for trained tokenizer
        if not self.vocab:
            raise RuntimeError(
                "Vocabulary not loaded. Train tokenizer or load vocab files first.")

        try:
            text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
            return text_bytes.decode("utf-8", errors="replace")
        except KeyError as e:
            raise ValueError(f"Unknown token ID: {e}")

    def save(self, file_path: str) -> None:
        """
        Save the trained tokenizer to files.

        Args:
            file_path: Base path for saving (will create .vocab and .merges files)
        """
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer must be trained before saving")

        vocab_path = f"{file_path}.vocab"
        merges_path = f"{file_path}.merges"

        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("# BPE Vocabulary\n")
            for token_id, token_bytes in self.vocab.items():
                f.write(f"{token_id}\t{token_bytes.hex()}\n")

        # Save merges
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("# BPE Merges\n")
            for (token1, token2), merged_id in self.merges.items():
                f.write(f"{token1}\t{token2}\t{merged_id}\n")

        logger.info(f"Tokenizer saved to {vocab_path} and {merges_path}")

    def load(self, file_path: str) -> None:
        """
        Load a trained tokenizer from files.

        Args:
            file_path: Base path for loading (expects .vocab and .merges files)
        """
        vocab_path = f"{file_path}.vocab"
        merges_path = f"{file_path}.merges"

        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError(
                f"Tokenizer files not found: {vocab_path}, {merges_path}")

        # Load vocabulary
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        token_id, token_hex = parts
                        self.vocab[int(token_id)] = bytes.fromhex(token_hex)

        # Load merges
        self.merges = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 3:
                        token1, token2, merged_id = parts
                        self.merges[(int(token1), int(token2))] = int(merged_id)

        logger.info(f"Tokenizer loaded from {vocab_path} and {merges_path}")
        logger.info(f"Vocabulary size: {len(self.vocab)}, Merges: {len(self.merges)}")

    def load_vocab_files(self, encoder_file: str, decoder_file: str) -> None:
        """
        Load vocabulary from encoder and decoder files.

        Args:
            encoder_file: Path to the encoder file (token -> id mapping)
            decoder_file: Path to the decoder file (id -> token mapping)
        """
        # Check if files exist
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Encoder file not found: {encoder_file}")
        if not os.path.exists(decoder_file):
            raise FileNotFoundError(f"Decoder file not found: {decoder_file}")
        # Load encoder
        self.encoder = {}
        try:
            with open(encoder_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')  # Only strip newlines, keep other spaces
                    if line:
                        try:
                            # Split from the right, only once, to separate token from ID
                            parts = line.rsplit(' ', 1)
                            if len(parts) == 2:
                                token, token_id = parts

                                # Handle escape sequences for special characters
                                token = token.replace('\\n', '\n').replace(
                                    '\\t', '\t').replace('\\\\', '\\')

                                self.encoder[token] = int(token_id)
                            else:
                                raise ValueError("Invalid format")
                        except ValueError:
                            raise ValueError(
                                f"Invalid encoder format in {encoder_file}: {line}")
        except UnicodeDecodeError:
            raise ValueError(f"Invalid encoder format in {encoder_file}")

        # Load decoder
        self.decoder = {}
        try:
            with open(decoder_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n\r')  # Only strip newlines, keep other spaces
                    if line:
                        try:
                            # Split from the left, only once, to separate ID from token
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                token_id, token = parts

                                # Handle escape sequences for special characters
                                token = token.replace('\\n', '\n').replace(
                                    '\\t', '\t').replace('\\\\', '\\')

                                self.decoder[int(token_id)] = token
                            else:
                                raise ValueError("Invalid format")
                        except ValueError:
                            raise ValueError(
                                f"Invalid decoder format in {decoder_file}: {line}")
        except UnicodeDecodeError:
            raise ValueError(f"Invalid decoder format in {decoder_file}")

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.encoder)

    @property
    def is_loaded(self) -> bool:
        """Return True if vocabulary is loaded."""
        return len(self.encoder) > 0

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
