"""Tests for the instruction dataset module."""
import json
import pytest
import torch
from pathlib import Path
from typing import Dict, List

from src.data import InstructionDataset, AlpacaDataset, create_instruction_dataset
from src.tokenizer import BPETokenizer


class TestInstructionDataset:
    """Test cases for InstructionDataset class."""

    @pytest.fixture
    def sample_instruction_data(self) -> List[Dict[str, str]]:
        """Create sample instruction data for testing."""
        return [
            {
                "instruction": "You are a helpful assistant.",
                "input": "What is the capital of France?",
                "output": "The capital of France is Paris."
            },
            {
                "instruction": "Answer questions about geography.",
                "input": "What is the largest country by area?",
                "output": "Russia is the largest country by area."
            },
            {
                "instruction": "Provide factual information.",
                "input": "",
                "output": "I can help you with factual information."
            }
        ]

    @pytest.fixture
    def instruction_file(self, temp_dir: Path, sample_instruction_data: List[Dict[str, str]]) -> Path:
        """Create a test instruction dataset file."""
        file_path = temp_dir / "test_instructions.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_instruction_data, f, indent=2)
        return file_path

    @pytest.fixture
    def tokenizer(self, vocab_files: Dict[str, Path]) -> BPETokenizer:
        """Create a tokenizer for testing."""
        return BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

    def test_instruction_dataset_initialization(
        self, 
        instruction_file: Path, 
        tokenizer: BPETokenizer,
        sample_instruction_data: List[Dict[str, str]]
    ):
        """Test InstructionDataset initialization."""
        dataset = InstructionDataset(
            data_path=instruction_file,
            tokenizer=tokenizer,
            max_length=64
        )

        assert dataset.data_path == instruction_file
        assert dataset.tokenizer is tokenizer
        assert dataset.max_length == 64
        assert len(dataset.data) == len(sample_instruction_data)
        assert len(dataset) == len(sample_instruction_data)

    def test_load_dataset_validation(
        self, 
        temp_dir: Path, 
        tokenizer: BPETokenizer
    ):
        """Test dataset loading with validation."""
        # Test with missing required fields
        invalid_data = [
            {"instruction": "Test", "output": "Response"}  # Missing input
        ]
        
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="Missing required field 'input'"):
            InstructionDataset(invalid_file, tokenizer)

    def test_load_dataset_nonexistent_file(self, tokenizer: BPETokenizer):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            InstructionDataset("/nonexistent/path.json", tokenizer)

    def test_load_dataset_invalid_json(self, temp_dir: Path, tokenizer: BPETokenizer):
        """Test loading invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            InstructionDataset(invalid_file, tokenizer)

    def test_process_dataset(
        self, 
        instruction_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test dataset processing."""
        dataset = InstructionDataset(
            data_path=instruction_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Check processed data structure
        assert len(dataset.processed_data) == len(dataset.data)
        
        for processed_item in dataset.processed_data:
            assert 'input_ids' in processed_item
            assert 'target_ids' in processed_item
            assert 'attention_mask' in processed_item
            assert 'labels_mask' in processed_item
            
            # Check tensor types and shapes
            assert isinstance(processed_item['input_ids'], torch.Tensor)
            assert isinstance(processed_item['target_ids'], torch.Tensor)
            assert isinstance(processed_item['attention_mask'], torch.Tensor)
            assert isinstance(processed_item['labels_mask'], torch.Tensor)
            
            # Check that tensors have correct length
            assert len(processed_item['input_ids']) <= dataset.max_length
            assert len(processed_item['target_ids']) <= dataset.max_length
            assert len(processed_item['attention_mask']) <= dataset.max_length
            assert len(processed_item['labels_mask']) <= dataset.max_length

    def test_getitem(self, instruction_file: Path, tokenizer: BPETokenizer):
        """Test dataset __getitem__ method."""
        dataset = InstructionDataset(
            data_path=instruction_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Test valid indices
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'target_ids' in item
        assert 'attention_mask' in item
        assert 'labels_mask' in item

        # Test index out of range
        with pytest.raises(IndexError):
            dataset[len(dataset)]

    def test_collate_fn(self, instruction_file: Path, tokenizer: BPETokenizer):
        """Test collate function for batching."""
        dataset = InstructionDataset(
            data_path=instruction_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Create a batch
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = dataset.collate_fn(batch)

        # Check batch structure
        assert 'input_ids' in collated
        assert 'target_ids' in collated
        assert 'attention_mask' in collated
        assert 'labels_mask' in collated

        # Check batch dimensions
        batch_size = len(batch)
        max_len = max(len(item['input_ids']) for item in batch)
        
        assert collated['input_ids'].shape == (batch_size, max_len)
        assert collated['target_ids'].shape == (batch_size, max_len)
        assert collated['attention_mask'].shape == (batch_size, max_len)
        assert collated['labels_mask'].shape == (batch_size, max_len)

    def test_template_formatting(
        self, 
        instruction_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test instruction template formatting."""
        custom_template = "Instruction: {instruction}\nInput: {input}\nResponse: "
        
        dataset = InstructionDataset(
            data_path=instruction_file,
            tokenizer=tokenizer,
            max_length=64,
            instruction_template=custom_template
        )

        assert dataset.instruction_template == custom_template

    def test_empty_input_handling(
        self, 
        temp_dir: Path, 
        tokenizer: BPETokenizer
    ):
        """Test handling of empty input fields."""
        data_with_empty_input = [
            {
                "instruction": "Provide help.",
                "input": "",
                "output": "I can help you."
            }
        ]
        
        file_path = temp_dir / "empty_input.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_with_empty_input, f)
        
        dataset = InstructionDataset(
            data_path=file_path,
            tokenizer=tokenizer,
            max_length=64
        )

        # Should handle empty input gracefully
        assert len(dataset) == 1
        item = dataset[0]
        assert isinstance(item, dict)


class TestAlpacaDataset:
    """Test cases for AlpacaDataset class."""

    @pytest.fixture
    def alpaca_data(self) -> List[Dict[str, str]]:
        """Create sample Alpaca format data."""
        return [
            {
                "instruction": "Classify the sentiment of this text.",
                "input": "I love this movie!",
                "output": "Positive"
            },
            {
                "instruction": "Translate to French.",
                "input": "Hello world",
                "output": "Bonjour le monde"
            }
        ]

    @pytest.fixture
    def alpaca_file(self, temp_dir: Path, alpaca_data: List[Dict[str, str]]) -> Path:
        """Create an Alpaca dataset file."""
        file_path = temp_dir / "alpaca_test.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, indent=2)
        return file_path

    @pytest.fixture
    def tokenizer(self, vocab_files: Dict[str, Path]) -> BPETokenizer:
        """Create a tokenizer for testing."""
        return BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

    def test_alpaca_dataset_initialization(
        self, 
        alpaca_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test AlpacaDataset initialization."""
        dataset = AlpacaDataset(
            data_path=alpaca_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Check that it uses correct Alpaca template
        expected_template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        assert dataset.instruction_template == expected_template
        assert isinstance(dataset, InstructionDataset)

    def test_alpaca_template_formatting(
        self, 
        alpaca_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test Alpaca template formatting."""
        dataset = AlpacaDataset(
            data_path=alpaca_file,
            tokenizer=tokenizer,
            max_length=128
        )

        # Process an item and check the formatted prompt
        item = dataset.data[0]
        formatted = dataset._format_prompt(item)
        
        # Should contain Alpaca-specific formatting
        assert "### Instruction:" in formatted
        assert "### Input:" in formatted
        assert "### Response:" in formatted
        assert item['instruction'] in formatted
        assert item['input'] in formatted


class TestCreateInstructionDataset:
    """Test cases for create_instruction_dataset factory function."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, str]]:
        """Create sample data for testing."""
        return [
            {
                "instruction": "Test instruction",
                "input": "Test input",
                "output": "Test output"
            }
        ]

    @pytest.fixture
    def test_file(self, temp_dir: Path, sample_data: List[Dict[str, str]]) -> Path:
        """Create test file."""
        file_path = temp_dir / "test.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f)
        return file_path

    @pytest.fixture
    def tokenizer(self, vocab_files: Dict[str, Path]) -> BPETokenizer:
        """Create a tokenizer for testing."""
        return BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

    def test_create_alpaca_dataset(
        self, 
        test_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test creating Alpaca dataset."""
        dataset = create_instruction_dataset(
            data_path=test_file,
            tokenizer=tokenizer,
            max_length=64,
            dataset_format="alpaca"
        )

        assert isinstance(dataset, AlpacaDataset)
        assert len(dataset) == 1

    def test_create_instruction_dataset_generic(
        self, 
        test_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test creating generic instruction dataset."""
        dataset = create_instruction_dataset(
            data_path=test_file,
            tokenizer=tokenizer,
            max_length=64,
            dataset_format="instruction"
        )

        assert isinstance(dataset, InstructionDataset)
        assert not isinstance(dataset, AlpacaDataset)
        assert len(dataset) == 1

    def test_create_dataset_unknown_format(
        self, 
        test_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test creating dataset with unknown format."""
        with pytest.raises(ValueError, match="Unknown dataset format"):
            create_instruction_dataset(
                data_path=test_file,
                tokenizer=tokenizer,
                max_length=64,
                dataset_format="unknown"
            )

    def test_create_dataset_default_format(
        self, 
        test_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test creating dataset with default format."""
        dataset = create_instruction_dataset(
            data_path=test_file,
            tokenizer=tokenizer,
            max_length=64
        )

        # Default should be alpaca
        assert isinstance(dataset, AlpacaDataset)


class TestInstructionDatasetErrorHandling:
    """Test error handling in instruction datasets."""

    @pytest.fixture
    def tokenizer(self, vocab_files: Dict[str, Path]) -> BPETokenizer:
        """Create a tokenizer for testing."""
        return BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

    def test_malformed_data_handling(
        self, 
        temp_dir: Path, 
        tokenizer: BPETokenizer
    ):
        """Test handling of malformed data."""
        # Test with non-list data
        malformed_file = temp_dir / "malformed.json"
        with open(malformed_file, 'w', encoding='utf-8') as f:
            json.dump({"not": "a list"}, f)
        
        with pytest.raises(ValueError, match="Expected a list of dictionaries"):
            InstructionDataset(malformed_file, tokenizer)

    def test_empty_dataset_handling(
        self, 
        temp_dir: Path, 
        tokenizer: BPETokenizer
    ):
        """Test handling of empty dataset."""
        empty_file = temp_dir / "empty.json"
        with open(empty_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        
        with pytest.raises(ValueError, match="Dataset is empty"):
            InstructionDataset(empty_file, tokenizer)

    def test_invalid_max_length(
        self, 
        temp_dir: Path, 
        tokenizer: BPETokenizer
    ):
        """Test handling of invalid max_length."""
        valid_data = [
            {
                "instruction": "Test",
                "input": "Test",
                "output": "Test"
            }
        ]
        
        test_file = temp_dir / "test.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(valid_data, f)
        
        with pytest.raises(ValueError, match="max_length must be positive"):
            InstructionDataset(test_file, tokenizer, max_length=0)
        
        with pytest.raises(ValueError, match="max_length must be positive"):
            InstructionDataset(test_file, tokenizer, max_length=-1)


class TestInstructionDatasetIntegration:
    """Integration tests for instruction datasets."""

    @pytest.fixture
    def real_dataset_file(self) -> Path:
        """Use the real example dataset."""
        return Path("example/fine-tuned/story_qa_dataset.json")

    @pytest.fixture
    def tokenizer(self, vocab_files: Dict[str, Path]) -> BPETokenizer:
        """Create a tokenizer for testing."""
        return BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

    @pytest.mark.integration
    def test_real_dataset_loading(
        self, 
        real_dataset_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test loading the real example dataset."""
        if not real_dataset_file.exists():
            pytest.skip("Real dataset file not found")
        
        dataset = create_instruction_dataset(
            data_path=real_dataset_file,
            tokenizer=tokenizer,
            max_length=256,
            dataset_format="alpaca"
        )

        assert len(dataset) > 0
        
        # Test that we can get items from the dataset
        item = dataset[0]
        assert isinstance(item, dict)
        assert all(key in item for key in ['input_ids', 'target_ids', 'attention_mask', 'labels_mask'])

    @pytest.mark.integration
    def test_dataset_with_dataloader(
        self, 
        real_dataset_file: Path, 
        tokenizer: BPETokenizer
    ):
        """Test dataset integration with DataLoader."""
        if not real_dataset_file.exists():
            pytest.skip("Real dataset file not found")
        
        from torch.utils.data import DataLoader
        
        dataset = create_instruction_dataset(
            data_path=real_dataset_file,
            tokenizer=tokenizer,
            max_length=256,
            dataset_format="alpaca"
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

        # Test that we can iterate through batches
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert all(key in batch for key in ['input_ids', 'target_ids', 'attention_mask', 'labels_mask'])
            
            # Check batch dimensions
            batch_size = batch['input_ids'].shape[0]
            assert batch_size <= 2
            
            # Check that all tensors have the same batch size
            for tensor in batch.values():
                assert tensor.shape[0] == batch_size
            
            break  # Just test the first batch
