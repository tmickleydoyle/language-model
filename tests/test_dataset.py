"""Tests for the dataset module."""
import pytest
import torch

from src.data import TextDataset
from src.tokenizer import BPETokenizer


class TestTextDataset:
    """Test cases for TextDataset class."""

    def test_dataset_initialization_with_text(self, sample_text_data, vocab_files):
        """Test TextDataset initialization with text data."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=16
        )

        assert dataset.block_size == 16
        assert dataset.tokenizer is tokenizer
        assert len(dataset.tokens) > 0
        assert len(dataset) > 0

    def test_dataset_initialization_with_file(
            self, temp_dir, sample_text_data, vocab_files):
        """Test TextDataset initialization with file path."""
        # Create text file
        text_file = temp_dir / "sample.txt"
        text_file.write_text(sample_text_data)

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            file_path=text_file,
            tokenizer=tokenizer,
            block_size=16
        )

        assert dataset.block_size == 16
        assert len(dataset.tokens) > 0
        assert len(dataset) > 0

    def test_dataset_initialization_no_data(self, vocab_files):
        """Test that providing no data raises ValueError."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        with pytest.raises(ValueError, match="Either text or file_path must be provided"):
            TextDataset(tokenizer=tokenizer, block_size=16)

    def test_dataset_initialization_both_data_sources(
            self, temp_dir, sample_text_data, vocab_files):
        """Test that providing both text and file raises ValueError."""
        text_file = temp_dir / "sample.txt"
        text_file.write_text(sample_text_data)

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        with pytest.raises(ValueError, match="Provide either text or file_path, not both"):
            TextDataset(
                text=sample_text_data,
                file_path=text_file,
                tokenizer=tokenizer,
                block_size=16
            )

    def test_dataset_nonexistent_file(self, vocab_files):
        """Test that non-existent file raises FileNotFoundError."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        with pytest.raises(FileNotFoundError):
            TextDataset(
                file_path="nonexistent.txt",
                tokenizer=tokenizer,
                block_size=16
            )

    def test_dataset_invalid_block_size(self, sample_text_data, vocab_files):
        """Test that invalid block_size raises ValueError."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        with pytest.raises(ValueError, match="block_size must be positive"):
            TextDataset(
                text=sample_text_data,
                tokenizer=tokenizer,
                block_size=0
            )

        with pytest.raises(ValueError, match="block_size must be positive"):
            TextDataset(
                text=sample_text_data,
                tokenizer=tokenizer,
                block_size=-1
            )

    def test_dataset_getitem(self, sample_text_data, vocab_files):
        """Test TextDataset __getitem__ method."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        block_size = 8
        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=block_size
        )

        # Test getting first item
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

        # y should be x shifted by one position
        assert torch.equal(y[:-1], x[1:])

    def test_dataset_getitem_last_index(self, sample_text_data, vocab_files):
        """Test getting last valid index."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        last_idx = len(dataset) - 1
        x, y = dataset[last_idx]

        assert x.shape == (8,)
        assert y.shape == (8,)

    def test_dataset_getitem_out_of_bounds(self, sample_text_data, vocab_files):
        """Test that out of bounds index raises IndexError."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]

        with pytest.raises(IndexError):
            _ = dataset[-1]  # Negative indexing not supported

    def test_dataset_len(self, sample_text_data, vocab_files):
        """Test TextDataset __len__ method."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        block_size = 8
        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=block_size
        )

        expected_len = len(dataset.tokens) - block_size
        assert len(dataset) == expected_len
        assert len(dataset) > 0

    def test_dataset_empty_text(self, vocab_files):
        """Test dataset with empty text."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text="",
            tokenizer=tokenizer,
            block_size=8
        )

        # Should handle empty text gracefully
        assert len(dataset.tokens) == 0
        assert len(dataset) == 0

    def test_dataset_short_text(self, vocab_files):
        """Test dataset with text shorter than block_size."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        short_text = "hello"
        block_size = 20  # Larger than tokenized text

        dataset = TextDataset(
            text=short_text,
            tokenizer=tokenizer,
            block_size=block_size
        )

        # Should handle short text gracefully
        if len(dataset.tokens) <= block_size:
            assert len(dataset) == 0
        else:
            assert len(dataset) > 0

    def test_dataset_tokenization_consistency(self, sample_text_data, vocab_files):
        """Test that tokenization is consistent."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Create dataset twice with same data
        dataset1 = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        dataset2 = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        # Should produce identical tokens
        assert torch.equal(dataset1.tokens, dataset2.tokens)
        assert len(dataset1) == len(dataset2)

    def test_dataset_different_block_sizes(self, sample_text_data, vocab_files):
        """Test dataset with different block sizes."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        block_sizes = [4, 8, 16, 32]
        datasets = []

        for block_size in block_sizes:
            dataset = TextDataset(
                text=sample_text_data,
                tokenizer=tokenizer,
                block_size=block_size
            )
            datasets.append(dataset)

            # Check that samples have correct size
            if len(dataset) > 0:
                x, y = dataset[0]
                assert x.shape == (block_size,)
                assert y.shape == (block_size,)

        # All datasets should have same tokens (different block sizes shouldn't
        # affect tokenization)
        for i in range(1, len(datasets)):
            assert torch.equal(datasets[0].tokens, datasets[i].tokens)

    def test_dataset_iteration(self, sample_text_data, vocab_files):
        """Test iterating over dataset."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        # Test iteration
        items = []
        for i in range(min(5, len(dataset))):  # Test first 5 items or all if less
            x, y = dataset[i]
            items.append((x, y))

            assert x.shape == (8,)
            assert y.shape == (8,)
            assert torch.equal(y[:-1], x[1:])

    def test_dataset_with_dataloader(self, sample_text_data, vocab_files):
        """Test dataset compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test getting batch
        for batch_x, batch_y in dataloader:
            assert batch_x.shape[0] <= 2  # Batch size
            assert batch_x.shape[1] == 8   # Block size
            assert batch_y.shape[0] <= 2   # Batch size
            assert batch_y.shape[1] == 8   # Block size
            break  # Just test first batch

    def test_dataset_repr(self, sample_text_data, vocab_files):
        """Test string representation of dataset."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=8
        )

        repr_str = repr(dataset)
        assert "TextDataset" in repr_str
        assert str(len(dataset)) in repr_str
        assert "8" in repr_str  # block_size

    def test_dataset_with_special_characters(self, vocab_files):
        """Test dataset with special characters in text."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        special_text = "Hello\nWorld\tTest\r\n!@#$%^&*()"

        dataset = TextDataset(
            text=special_text,
            tokenizer=tokenizer,
            block_size=8
        )

        # Should handle special characters without errors
        assert len(dataset.tokens) >= 0
        if len(dataset) > 0:
            x, y = dataset[0]
            assert x.shape == (8,)
            assert y.shape == (8,)

    def test_dataset_large_block_size(self, sample_text_data, vocab_files):
        """Test dataset with very large block size."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        large_block_size = 1000  # Much larger than typical text

        dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=large_block_size
        )

        # If text is shorter than block size, dataset should be empty
        if len(dataset.tokens) <= large_block_size:
            assert len(dataset) == 0
        else:
            assert len(dataset) > 0
            x, y = dataset[0]
            assert x.shape == (large_block_size,)
            assert y.shape == (large_block_size,)

    @pytest.mark.parametrize("encoding", ["utf-8", "latin1"])
    def test_dataset_different_encodings(
            self,
            temp_dir,
            sample_text_data,
            vocab_files,
            encoding):
        """Test dataset with different file encodings."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Create file with specific encoding
        text_file = temp_dir / f"sample_{encoding}.txt"
        text_file.write_text(sample_text_data, encoding=encoding)

        dataset = TextDataset(
            file_path=text_file,
            tokenizer=tokenizer,
            block_size=8
        )

        assert len(dataset.tokens) > 0
        assert len(dataset) >= 0
