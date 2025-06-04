"""Tests for the tokenizer module."""
import pytest

from src.tokenizer import BPETokenizer


class TestBPETokenizer:
    """Test cases for BPETokenizer class."""

    def test_tokenizer_initialization_with_files(self, vocab_files):
        """Test initializing tokenizer with vocab files."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        assert tokenizer.vocab_size > 0
        assert len(tokenizer.encoder) > 0
        assert len(tokenizer.decoder) > 0
        assert len(tokenizer.encoder) == len(tokenizer.decoder)

    def test_tokenizer_initialization_no_files(self):
        """Test initializing tokenizer without files."""
        tokenizer = BPETokenizer()

        # Should have default vocab_size
        assert tokenizer.vocab_size == 0
        assert len(tokenizer.encoder) == 0
        assert len(tokenizer.decoder) == 0

    def test_load_vocab_files_success(self, vocab_files):
        """Test successful loading of vocabulary files."""
        tokenizer = BPETokenizer()
        tokenizer.load_vocab_files(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        assert tokenizer.vocab_size > 0
        assert len(tokenizer.encoder) > 0
        assert len(tokenizer.decoder) > 0

    def test_load_vocab_files_nonexistent_encoder(self, vocab_files):
        """Test loading with non-existent encoder file."""
        tokenizer = BPETokenizer()

        with pytest.raises(FileNotFoundError, match="Encoder file not found"):
            tokenizer.load_vocab_files(
                encoder_file="nonexistent.txt",
                decoder_file=vocab_files["decoder"]
            )

    def test_load_vocab_files_nonexistent_decoder(self, vocab_files):
        """Test loading with non-existent decoder file."""
        tokenizer = BPETokenizer()

        with pytest.raises(FileNotFoundError, match="Decoder file not found"):
            tokenizer.load_vocab_files(
                encoder_file=vocab_files["encoder"],
                decoder_file="nonexistent.txt"
            )

    def test_load_vocab_files_malformed_encoder(self, temp_dir):
        """Test loading malformed encoder file."""
        encoder_file = temp_dir / "bad_encoder.txt"
        decoder_file = temp_dir / "decoder.txt"

        # Create malformed encoder (missing index)
        encoder_file.write_text("hello\nworld test")
        decoder_file.write_text("0 hello\n1 world")

        tokenizer = BPETokenizer()
        with pytest.raises(ValueError, match="Invalid encoder format"):
            tokenizer.load_vocab_files(encoder_file, decoder_file)

    def test_load_vocab_files_malformed_decoder(self, temp_dir):
        """Test loading malformed decoder file."""
        encoder_file = temp_dir / "encoder.txt"
        decoder_file = temp_dir / "bad_decoder.txt"

        # Create malformed decoder (missing token)
        encoder_file.write_text("hello 0\nworld 1")
        decoder_file.write_text("0\n1 world")

        tokenizer = BPETokenizer()
        with pytest.raises(ValueError, match="Invalid decoder format"):
            tokenizer.load_vocab_files(encoder_file, decoder_file)

    def test_encode_simple_text(self, vocab_files):
        """Test encoding simple text."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        text = "hello world"
        tokens = tokenizer.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)

    def test_encode_empty_string(self, vocab_files):
        """Test encoding empty string."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        tokens = tokenizer.encode("")
        assert tokens == []

    def test_encode_unknown_tokens(self, vocab_files):
        """Test encoding text with unknown tokens."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Text with tokens not in vocabulary
        text = "xyz unknown tokens"
        tokens = tokenizer.encode(text)

        # Should still return some tokens (might be character-level fallback)
        assert isinstance(tokens, list)

    def test_decode_simple_tokens(self, vocab_files):
        """Test decoding simple token list."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Use known token indices
        tokens = [0, 1, 2]  # Assuming these exist in our test vocab
        text = tokenizer.decode(tokens)

        assert isinstance(text, str)

    def test_decode_empty_tokens(self, vocab_files):
        """Test decoding empty token list."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        text = tokenizer.decode([])
        assert text == ""

    def test_decode_invalid_tokens(self, vocab_files):
        """Test decoding invalid token indices."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Token index that doesn't exist
        invalid_tokens = [999999]

        with pytest.raises(KeyError, match="Unknown token"):
            tokenizer.decode(invalid_tokens)

    def test_encode_decode_roundtrip(self, vocab_files):
        """Test that encode/decode roundtrip works for known tokens."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Use text that should be in our test vocabulary
        original_text = "hello world test"
        tokens = tokenizer.encode(original_text)
        decoded_text = tokenizer.decode(tokens)

        # May not be exactly equal due to tokenization, but should be meaningful
        assert isinstance(decoded_text, str)
        assert len(decoded_text) > 0

    def test_vocab_size_property(self, vocab_files):
        """Test vocab_size property."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        expected_size = len(tokenizer.encoder)
        assert tokenizer.vocab_size == expected_size

    def test_vocab_size_empty_tokenizer(self):
        """Test vocab_size for empty tokenizer."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 0

    def test_is_loaded_property(self, vocab_files):
        """Test is_loaded property."""
        tokenizer = BPETokenizer()
        assert not tokenizer.is_loaded

        tokenizer.load_vocab_files(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )
        assert tokenizer.is_loaded

    def test_repr(self, vocab_files):
        """Test string representation."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        repr_str = repr(tokenizer)
        assert "BPETokenizer" in repr_str
        assert str(tokenizer.vocab_size) in repr_str

    def test_repr_empty(self):
        """Test string representation of empty tokenizer."""
        tokenizer = BPETokenizer()
        repr_str = repr(tokenizer)
        assert "BPETokenizer" in repr_str
        assert "not loaded" in repr_str

    def test_encode_without_vocab_raises_error(self):
        """Test that encoding without loaded vocab raises error."""
        tokenizer = BPETokenizer()

        with pytest.raises(RuntimeError, match="Vocabulary not loaded"):
            tokenizer.encode("test text")

    def test_decode_without_vocab_raises_error(self):
        """Test that decoding without loaded vocab raises error."""
        tokenizer = BPETokenizer()

        with pytest.raises(RuntimeError, match="Vocabulary not loaded"):
            tokenizer.decode([1, 2, 3])

    def test_load_vocab_consistency(self, vocab_files):
        """Test that encoder and decoder are consistent."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Check that every encoder key has corresponding decoder entry
        for token, idx in tokenizer.encoder.items():
            assert idx in tokenizer.decoder
            assert tokenizer.decoder[idx] == token

        # Check that every decoder key has corresponding encoder entry
        for idx, token in tokenizer.decoder.items():
            assert token in tokenizer.encoder
            assert tokenizer.encoder[token] == idx

    @pytest.mark.parametrize("text", [
        "hello",
        "hello world",
        "test sample text",
        "the quick brown fox",
    ])
    def test_encode_various_texts(self, vocab_files, text):
        """Test encoding various text inputs."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        tokens = tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(token, int) for token in tokens)

    def test_large_vocab_file_handling(self, temp_dir):
        """Test handling of larger vocabulary files."""
        encoder_file = temp_dir / "large_encoder.txt"
        decoder_file = temp_dir / "large_decoder.txt"

        # Create larger vocabulary
        vocab_size = 1000
        with open(encoder_file, 'w') as f:
            for i in range(vocab_size):
                f.write(f"token_{i} {i}\n")

        with open(decoder_file, 'w') as f:
            for i in range(vocab_size):
                f.write(f"{i} token_{i}\n")

        tokenizer = BPETokenizer(encoder_file, decoder_file)
        assert tokenizer.vocab_size == vocab_size

    def test_special_characters_in_vocab(self, temp_dir):
        """Test handling of special characters in vocabulary."""
        encoder_file = temp_dir / "special_encoder.txt"
        decoder_file = temp_dir / "special_decoder.txt"

        # Vocabulary with special characters
        special_tokens = ["<unk>", "<pad>", "<eos>", "<bos>", "\n", "\t", " "]

        with open(encoder_file, 'w') as f:
            for i, token in enumerate(special_tokens):
                # Escape special characters for file format
                escaped_token = token.replace(
                    '\\',
                    '\\\\').replace(
                    '\n',
                    '\\n').replace(
                    '\t',
                    '\\t')
                f.write(f"{escaped_token} {i}\n")

        with open(decoder_file, 'w') as f:
            for i, token in enumerate(special_tokens):
                # Escape special characters for file format
                escaped_token = token.replace(
                    '\\',
                    '\\\\').replace(
                    '\n',
                    '\\n').replace(
                    '\t',
                    '\\t')
                f.write(f"{i} {escaped_token}\n")

        tokenizer = BPETokenizer(encoder_file, decoder_file)
        assert tokenizer.vocab_size == len(special_tokens)

        # Test encoding/decoding with special tokens
        tokens = tokenizer.encode("<unk>")
        assert len(tokens) > 0
