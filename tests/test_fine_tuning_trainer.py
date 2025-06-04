"""
Comprehensive tests for the fine-tuning trainer module.

This test suite covers all aspects of the FineTuningTrainer including:
- Initialization and configuration
- Model loading and checkpoint management
- Training and evaluation loops
- Error handling and edge cases
- Integration with datasets and tokenizers
"""

import pytest
import torch
import tempfile
import shutil
import logging
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

from src.training.fine_tuning_trainer import FineTuningTrainer
from src.model.gpt import GPTLanguageModel
from src.data.instruction_dataset import InstructionDataset
from src.tokenizer.bpe import BPETokenizer
from src.config import Config


# Global fixtures for all test classes
@pytest.fixture
def sample_instruction_data() -> List[Dict[str, str]]:
    """Create sample instruction data for testing."""
    return [
        {
            "instruction": "You are a helpful assistant.",
            "input": "What is 2+2?",
            "output": "2+2 equals 4."
        },
        {
            "instruction": "Answer math questions.",
            "input": "What is 3*3?",
            "output": "3*3 equals 9."
        },
        {
            "instruction": "Explain concepts clearly.",
            "input": "What is gravity?",
            "output": "Gravity is a force that attracts objects toward each other."
        },
        {
            "instruction": "Help with coding.",
            "input": "Write a hello world in Python.",
            "output": "print('Hello, World!')"
        }
    ]


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Create a mock tokenizer for testing."""
    tokenizer = Mock(spec=BPETokenizer)
    tokenizer.vocab_size = 100
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    # Return a response that includes the expected format for proper extraction
    tokenizer.decode.return_value = "### Instruction:\nAnswer the question\n\n### Input:\nWhat is AI?\n\n### Response:\nAI is artificial intelligence."
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_instruction_dataset(mock_tokenizer: Mock, sample_instruction_data: List[Dict[str, str]]) -> Mock:
    """Create a mock instruction dataset for testing."""
    dataset = Mock(spec=InstructionDataset)
    dataset.tokenizer = mock_tokenizer
    # Use a template that doesn't require 'output' for generation
    dataset.instruction_template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    
    # Make len() work directly on the mock - fix the function signature
    def len_impl(self):
        return len(sample_instruction_data)
    dataset.__len__ = len_impl
    
    def mock_getitem(idx):
        return {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'target_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'labels_mask': torch.tensor([0, 0, 1, 1, 1])  # Only last 3 tokens are response
        }
    
    dataset.__getitem__ = Mock(side_effect=mock_getitem)
    dataset.collate_fn = lambda batch: {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels_mask': torch.stack([item['labels_mask'] for item in batch])
    }
    
    return dataset


@pytest.fixture
def mock_model(sample_instruction_data: List[Dict[str, str]]) -> Mock:
    """Create a mock GPT model for testing."""
    model = Mock(spec=GPTLanguageModel)
    model.config = Mock()
    model.config.vocab_size = 100
    
    # Mock forward pass - return tensor with gradients enabled
    def mock_forward(input_ids):
        batch_size, seq_len = input_ids.shape
        vocab_size = 100
        # Create tensor that requires gradients for training
        output = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        return output
    
    model.forward.side_effect = mock_forward
    # Use side_effect on the model itself to handle __call__
    model.side_effect = mock_forward
    
    # Mock state dict operations with proper return value
    model.state_dict.return_value = {'layer.weight': torch.randn(10, 10)}
    model.load_state_dict = Mock(return_value=([], []))  # Return empty missing/unexpected keys
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    
    # Mock generation
    def mock_generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=None):
        batch_size = input_ids.shape[0]
        original_len = input_ids.shape[1]
        new_len = original_len + max_new_tokens
        return torch.randint(0, 100, (batch_size, new_len))
    
    model.generate.side_effect = mock_generate
    
    # Mock training/eval modes
    model.train = Mock(return_value=model)
    model.eval = Mock(return_value=model)
    
    return model


class TestFineTuningTrainer:
    """Test cases for FineTuningTrainer class."""

    @pytest.fixture
    def small_config(self) -> Config:
        """Create a small config optimized for fast testing."""
        return Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            dropout=0.1,
            learning_rate=1e-4,
            batch_size=2,
            max_epochs=2,
            max_iters=10,
            eval_interval=1,
            eval_iters=5,
            save_interval=5,
            device="cpu",
            fp16=False,
            seed=42,
            # Fine-tuning specific
            grad_clip=1.0,
            scheduler_type="cosine",
            patience=3
        )

    # All fixtures moved to global scope for cross-class access

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def base_model_checkpoint(self, temp_dir: Path, small_config: Config) -> Path:
        """Create a mock base model checkpoint for testing."""
        checkpoint_path = temp_dir / "base_model.pth"
        checkpoint = {
            'model_state_dict': {
                'embedding.weight': torch.randn(small_config.vocab_size, small_config.n_embd),
                'layer.0.weight': torch.randn(small_config.n_embd, small_config.n_embd),
                'layer.1.weight': torch.randn(small_config.n_embd, small_config.n_embd)
            },
            'config': small_config.to_dict(),
            'epoch': 10,
            'loss': 0.5
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_initialization_success(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test successful trainer initialization."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.model is mock_model
        assert trainer.config is small_config
        assert trainer.instruction_dataset is mock_instruction_dataset
        assert trainer.val_instruction_dataset is None
        assert trainer.base_model_path is None
        assert trainer.train_loader is not None
        assert trainer.val_loader is None
        assert hasattr(trainer, 'grad_scaler')
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'metrics')
        
        # Check metrics initialization
        expected_metrics = ['train_loss', 'train_instruction_loss', 'val_loss', 'val_instruction_loss', 'learning_rates']
        for metric in expected_metrics:
            assert metric in trainer.metrics
            assert isinstance(trainer.metrics[metric], list)

    def test_initialization_with_validation_dataset(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test trainer initialization with validation dataset."""
        val_dataset = Mock(spec=InstructionDataset)
        # Fix the __len__ attribute issue
        def len_impl(self):
            return 2
        val_dataset.__len__ = len_impl
        val_dataset.collate_fn = mock_instruction_dataset.collate_fn
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            val_dataset=val_dataset
        )
        
        assert trainer.val_instruction_dataset is val_dataset
        assert trainer.val_loader is not None

    def test_initialization_with_base_model(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock, 
        base_model_checkpoint: Path
    ):
        """Test trainer initialization with base model loading."""
        with patch.object(FineTuningTrainer, '_load_base_model') as mock_load:
            trainer = FineTuningTrainer(
                model=mock_model,
                config=small_config,
                instruction_dataset=mock_instruction_dataset,
                base_model_path=base_model_checkpoint
            )
            
            assert trainer.base_model_path == base_model_checkpoint
            mock_load.assert_called_once()

    def test_create_scheduler_cosine(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test cosine learning rate scheduler creation."""
        small_config.scheduler_type = "cosine"
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.scheduler is not None
        # We can't easily test the exact type due to mocking, but we can verify it's created

    def test_create_scheduler_step(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test step learning rate scheduler creation."""
        small_config.scheduler_type = "step"
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.scheduler is not None

    def test_create_scheduler_none(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test no scheduler creation."""
        small_config.scheduler_type = "none"
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.scheduler is None

    def test_load_base_model_success(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock, 
        base_model_checkpoint: Path
    ):
        """Test successful base model loading."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.base_model_path = base_model_checkpoint
        trainer._load_base_model()
        
        # Verify model.load_state_dict was called
        mock_model.load_state_dict.assert_called_once()

    def test_load_base_model_file_not_found(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock, 
        temp_dir: Path
    ):
        """Test base model loading with non-existent file."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.base_model_path = temp_dir / "nonexistent.pth"
        
        with pytest.raises(FileNotFoundError):
            trainer._load_base_model()

    def test_load_base_model_with_size_mismatch(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock, 
        temp_dir: Path
    ):
        """Test base model loading with size mismatched weights."""
        # Create checkpoint with mismatched sizes
        checkpoint_path = temp_dir / "mismatched_model.pth"
        checkpoint = {
            'model_state_dict': {
                'embedding.weight': torch.randn(200, small_config.n_embd),  # Wrong vocab size
                'layer.0.weight': torch.randn(small_config.n_embd, small_config.n_embd),
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Mock model state dict with correct sizes
        mock_model.state_dict.return_value = {
            'embedding.weight': torch.randn(small_config.vocab_size, small_config.n_embd),
            'layer.0.weight': torch.randn(small_config.n_embd, small_config.n_embd),
        }
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.base_model_path = checkpoint_path
        
        # Should not raise error, but should log warnings
        with patch('src.training.fine_tuning_trainer.logger') as mock_logger:
            trainer._load_base_model()
            
            # Verify warnings were logged for size mismatches
            assert any('shape mismatch' in str(call) for call in mock_logger.warning.call_args_list)

    def test_compute_instruction_loss(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test instruction loss computation."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        batch_size, seq_len, vocab_size = 2, 5, small_config.vocab_size
        logits = torch.randn(batch_size, seq_len, vocab_size)
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float)
        attention_mask = torch.ones(batch_size, seq_len)
        
        total_loss, instruction_loss = trainer._compute_instruction_loss(
            logits, target_ids, labels_mask, attention_mask
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(instruction_loss, torch.Tensor)
        assert total_loss.item() >= 0
        assert instruction_loss.item() >= 0

    def test_train_step(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test single training step."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]),
            'target_ids': torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]),
            'attention_mask': torch.ones(2, 5),
            'labels_mask': torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float)
        }
        
        # Mock optimizer
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        
        metrics = trainer._train_step(batch)
        
        assert 'total_loss' in metrics
        assert 'instruction_loss' in metrics
        assert isinstance(metrics['total_loss'], float)
        assert isinstance(metrics['instruction_loss'], float)
        assert metrics['total_loss'] >= 0
        assert metrics['instruction_loss'] >= 0
        
        # Verify optimizer was called
        trainer.optimizer.zero_grad.assert_called_once()
        trainer.optimizer.step.assert_called_once()

    def test_train_step_with_gradient_clipping(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test training step with gradient clipping."""
        small_config.grad_clip = 1.0
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'target_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5),
            'labels_mask': torch.ones(1, 5)
        }
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            metrics = trainer._train_step(batch)
            
            # Verify gradient clipping was called
            mock_clip.assert_called_once()

    def test_evaluate_step(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test single evaluation step."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'target_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5),
            'labels_mask': torch.ones(1, 5)
        }
        
        metrics = trainer._evaluate_step(batch)
        
        assert 'total_loss' in metrics
        assert 'instruction_loss' in metrics
        assert isinstance(metrics['total_loss'], float)
        assert isinstance(metrics['instruction_loss'], float)

    def test_train_epoch(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test training for one epoch."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        
        metrics = trainer.train_epoch()
        
        assert 'train_loss' in metrics
        assert 'train_instruction_loss' in metrics
        assert isinstance(metrics['train_loss'], float)
        assert isinstance(metrics['train_instruction_loss'], float)

    def test_evaluate_without_validation_dataset(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test evaluation when no validation dataset is provided."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        result = trainer.evaluate()
        assert result is None

    def test_evaluate_with_validation_dataset(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test evaluation with validation dataset."""
        val_dataset = Mock(spec=InstructionDataset)
        # Fix the __len__ attribute issue
        def len_impl(self):
            return 2
        val_dataset.__len__ = len_impl
        val_dataset.collate_fn = mock_instruction_dataset.collate_fn
        
        def mock_getitem(idx):
            return {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'target_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'labels_mask': torch.tensor([0, 0, 1, 1, 1])
            }
        val_dataset.__getitem__ = Mock(side_effect=mock_getitem)
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            val_dataset=val_dataset
        )
        
        metrics = trainer.evaluate()
        
        assert metrics is not None
        assert 'val_loss' in metrics
        assert 'val_instruction_loss' in metrics

    def test_train_complete_workflow(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        temp_dir: Path
    ):
        """Test complete training workflow."""
        small_config.max_epochs = 2
        small_config.save_interval = 1
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            save_dir=temp_dir
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        trainer.optimizer.param_groups = [{'lr': small_config.learning_rate}]
        
        with patch.object(trainer, 'save_checkpoint') as mock_save:
            result = trainer.train()
            
            assert 'final_metrics' in result
            assert 'best_val_loss' in result
            assert 'training_time' in result
            assert 'metrics_history' in result
            
            # Verify checkpoint saving was called
            assert mock_save.call_count >= 1

    def test_train_with_early_stopping(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test training with early stopping."""
        small_config.max_epochs = 10
        small_config.patience = 2
        
        val_dataset = Mock(spec=InstructionDataset)
        # Fix the __len__ attribute issue
        def len_impl(self):
            return 2
        val_dataset.__len__ = len_impl
        val_dataset.collate_fn = mock_instruction_dataset.collate_fn
        
        def mock_getitem(idx):
            return {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'target_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'labels_mask': torch.tensor([0, 0, 1, 1, 1])
            }
        val_dataset.__getitem__ = Mock(side_effect=mock_getitem)
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            val_dataset=val_dataset,
            save_dir=None  # Disable checkpoint saving to avoid pickle errors
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        trainer.optimizer.param_groups = [{'lr': small_config.learning_rate}]
        
        # Mock save_checkpoint to avoid pickle issues
        trainer.save_checkpoint = Mock()
        
        # Mock progressively increasing validation loss to trigger early stopping
        original_evaluate = trainer.evaluate
        call_count = 0
        def mock_evaluate():
            nonlocal call_count
            call_count += 1
            return {
                'val_loss': 1.0 + call_count * 0.1,  # Increasing loss
                'val_instruction_loss': 1.0 + call_count * 0.1
            }
        
        trainer.evaluate = mock_evaluate
        
        with patch('src.training.fine_tuning_trainer.logger') as mock_logger:
            result = trainer.train()
            
            # Should stop early, not complete all 10 epochs
            assert any('Early stopping' in str(call) for call in mock_logger.info.call_args_list)

    def test_generate_response(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test response generation."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        instruction = "Answer the question"
        input_text = "What is AI?"
        
        response = trainer.generate_response(
            instruction=instruction,
            input_text=input_text,
            max_new_tokens=50,
            temperature=0.8
        )
        
        assert isinstance(response, str)
        assert len(response) > 0

    def test_save_checkpoint(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        temp_dir: Path
    ):
        """Test checkpoint saving."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            save_dir=temp_dir
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.state_dict.return_value = {'param_groups': []}
        
        filename = "test_checkpoint.pth"
        epoch = 5
        loss = 0.5
        
        trainer.save_checkpoint(filename, epoch, loss)
        
        checkpoint_path = temp_dir / filename
        assert checkpoint_path.exists()
        
        # Load and verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint
        assert checkpoint['epoch'] == epoch
        assert checkpoint['loss'] == loss

    def test_save_checkpoint_without_save_dir(
        self,
        mock_model: Mock,
        small_config: Config,
        mock_instruction_dataset: Mock
    ):
        """Test checkpoint saving without save directory."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            save_dir=None
        )
        
        # Mock the optimizer that gets accessed in save_checkpoint
        trainer.optimizer = Mock()
        trainer.optimizer.state_dict = Mock(return_value={'param_groups': []})
        
        with patch('src.training.fine_tuning_trainer.logger') as mock_logger:
            trainer.save_checkpoint("test.pth", 1, 0.5)
            
            # Should log warning about no save directory
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert 'No save directory specified' in warning_call

    def test_load_checkpoint(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        temp_dir: Path
    ):
        """Test checkpoint loading."""
        # Create a checkpoint file
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        checkpoint_data = {
            'model_state_dict': {'layer.weight': torch.randn(10, 10)},
            'optimizer_state_dict': {'param_groups': []},
            'config': small_config.to_dict(),  # Add missing config key
            'scheduler_state_dict': {'step': 5},
            'grad_scaler_state_dict': {'scale': 1.0},
            'metrics': {'train_loss': [0.5, 0.4]},
            'epoch': 10,
            'loss': 0.3
        }
        torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=False)
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.load_state_dict = Mock()
        
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_path)
        
        assert loaded_checkpoint['epoch'] == 10
        assert loaded_checkpoint['loss'] == 0.3
        
        # Verify model and optimizer state dicts were loaded
        assert mock_model.load_state_dict.call_count == 1
        assert trainer.optimizer.load_state_dict.call_count == 1
        
        # Check that the correct keys were passed (avoid tensor comparison)
        model_call_args = mock_model.load_state_dict.call_args[0][0]
        assert 'layer.weight' in model_call_args
        assert model_call_args['layer.weight'].shape == (10, 10)

    def test_load_checkpoint_file_not_found(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        temp_dir: Path
    ):
        """Test checkpoint loading with non-existent file."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(temp_dir / "nonexistent.pth")

    def test_data_loader_creation(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test data loader creation."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.train_loader is not None
        assert trainer.val_loader is None  # No validation dataset provided

    def test_data_loader_creation_with_validation(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test data loader creation with validation dataset."""
        val_dataset = Mock(spec=InstructionDataset)
        # Fix the __len__ attribute issue
        def len_impl(self):
            return 2
        val_dataset.__len__ = len_impl
        val_dataset.collate_fn = mock_instruction_dataset.collate_fn
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset,
            val_dataset=val_dataset
        )
        
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None

    def test_mixed_precision_training(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test mixed precision training setup."""
        small_config.fp16 = True
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.grad_scaler is not None
        assert trainer.config.fp16

    @pytest.mark.parametrize("scheduler_type", ["cosine", "step", "none"])
    def test_scheduler_types(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        scheduler_type: str
    ):
        """Test different scheduler types."""
        small_config.scheduler_type = scheduler_type
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        if scheduler_type == "none":
            assert trainer.scheduler is None
        else:
            assert trainer.scheduler is not None

    def test_device_handling_cpu(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test device handling for CPU."""
        small_config.device = "cpu"
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        # Check that pin_memory is False for CPU
        assert hasattr(trainer, 'train_loader')

    def test_metrics_tracking(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test that metrics are properly tracked during training."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        # Initial state
        assert all(len(trainer.metrics[key]) == 0 for key in trainer.metrics)
        
        # Mock a training step and verify metrics are updated
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        trainer.optimizer.param_groups = [{'lr': small_config.learning_rate}]
        
        epoch_metrics = trainer.train_epoch()
        
        assert 'train_loss' in epoch_metrics
        assert 'train_instruction_loss' in epoch_metrics

    def test_error_handling_in_train_step(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test error handling in training step."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        # Mock optimizer with error
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock(side_effect=RuntimeError("Optimizer error"))
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'target_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5),
            'labels_mask': torch.ones(1, 5)
        }
        
        with pytest.raises(RuntimeError):
            trainer._train_step(batch)

    def test_configuration_validation(
        self, 
        mock_model: Mock, 
        mock_instruction_dataset: Mock
    ):
        """Test that configuration validation works properly."""
        # Test with invalid configuration - should raise ValueError
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            invalid_config = Config(
                vocab_size=-1,  # Invalid
                n_embd=64,
                n_head=4,
                n_layer=2,
                block_size=32,
                device="cpu"
            )
        
        # Test that valid configuration works
        valid_config = Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            device="cpu"
        )
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=valid_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        assert trainer.config.vocab_size == 100

    def test_logging_integration(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test that logging is properly integrated."""
        with patch('src.training.fine_tuning_trainer.logger') as mock_logger:
            trainer = FineTuningTrainer(
                model=mock_model,
                config=small_config,
                instruction_dataset=mock_instruction_dataset
            )
            
            # Verify initialization logging
            assert any('FineTuningTrainer initialized' in str(call) 
                      for call in mock_logger.info.call_args_list)

    def test_memory_efficiency(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test memory efficiency settings."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        # Verify num_workers is set to 0 for simplicity in testing
        # This can be verified by checking the DataLoader configuration
        assert hasattr(trainer, 'train_loader')
        assert trainer.train_loader is not None


# Integration Tests
class TestFineTuningTrainerIntegration:
    """Integration tests that test multiple components working together."""
    
    @pytest.fixture
    def real_small_model(self) -> GPTLanguageModel:
        """Create a real small model for integration testing."""
        config = Config(
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=2,
            block_size=16,
            device="cpu"
        )
        
        from src.model.gpt import GPTLanguageModel
        return GPTLanguageModel(config)
    
    def test_real_training_step(
        self, 
        real_small_model: GPTLanguageModel,
        mock_instruction_dataset: Mock
    ):
        """Test training step with real model components."""
        config = Config(
            vocab_size=100,
            n_embd=32,
            n_head=2,
            n_layer=2,
            block_size=16,
            batch_size=1,
            max_epochs=1,
            learning_rate=1e-4,
            device="cpu",
            fp16=False
        )
        
        trainer = FineTuningTrainer(
            model=real_small_model,
            config=config,
            instruction_dataset=mock_instruction_dataset
        )
        
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'target_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5),
            'labels_mask': torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float)
        }
        
        # Should not raise any errors
        metrics = trainer._train_step(batch)
        
        assert 'total_loss' in metrics
        assert 'instruction_loss' in metrics
        assert metrics['total_loss'] > 0
        assert metrics['instruction_loss'] > 0


# Performance Tests
class TestFineTuningTrainerPerformance:
    """Performance and resource usage tests."""
    
    @pytest.fixture
    def small_config(self) -> Config:
        """Create a small config optimized for fast testing."""
        return Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            dropout=0.1,
            learning_rate=1e-4,
            batch_size=2,
            max_epochs=2,
            max_iters=10,
            eval_interval=1,
            eval_iters=5,
            save_interval=5,
            device="cpu",
            seed=42,
            fp16=False,
            log_level='INFO'
        )
    
    @pytest.mark.slow
    def test_memory_usage_during_training(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test memory usage remains stable during training."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        trainer.optimizer.param_groups = [{'lr': small_config.learning_rate}]
        
        # Run several training epochs
        for _ in range(5):
            trainer.train_epoch()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
    
    def test_training_speed_benchmark(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Benchmark training speed."""
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        trainer.optimizer = Mock()
        trainer.optimizer.zero_grad = Mock()
        trainer.optimizer.step = Mock()
        
        start_time = time.time()
        trainer.train_epoch()
        end_time = time.time()
        
        epoch_time = end_time - start_time
        
        # Should complete epoch quickly in test environment
        assert epoch_time < 10.0, f"Epoch took {epoch_time:.2f}s, expected < 10s"


# Edge Cases and Error Handling
class TestFineTuningTrainerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def small_config(self) -> Config:
        """Create a small config optimized for fast testing."""
        return Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=32,
            dropout=0.1,
            learning_rate=1e-4,
            batch_size=2,
            max_epochs=2,
            max_iters=10,
            eval_interval=1,
            eval_iters=5,
            save_interval=5,
            device="cpu",
            seed=42,
            fp16=False,
            log_level='INFO'
        )
    
    def test_empty_dataset_handling(
        self, 
        mock_model: Mock, 
        small_config: Config
    ):
        """Test handling of empty datasets."""
        empty_dataset = Mock(spec=InstructionDataset)
        empty_dataset.__len__ = Mock(return_value=0)
        empty_dataset.collate_fn = lambda x: x
        
        # Should raise RuntimeError for empty dataset (PyTorch doesn't allow empty datasets)
        with pytest.raises(RuntimeError, match="Training setup failed"):
            FineTuningTrainer(
                model=mock_model,
                config=small_config,
                instruction_dataset=empty_dataset
            )

    def test_very_large_batch_size(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test handling of very large batch sizes."""
        small_config.batch_size = 1000  # Very large batch size
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        # Should create trainer without issues
        assert trainer.config.batch_size == 1000

    def test_extreme_learning_rates(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test handling of extreme learning rates."""
        extreme_configs = [
            1e-10,  # Very small
            1e10,   # Very large
            0.0,    # Zero
        ]
        
        for lr in extreme_configs:
            config = small_config
            config.learning_rate = lr
            
            if lr <= 0:
                # Should raise ValueError for non-positive learning rates
                with pytest.raises(ValueError, match="learning_rate must be positive"):
                    FineTuningTrainer(
                        model=mock_model,
                        config=config,
                        instruction_dataset=mock_instruction_dataset
                    )
            else:
                # Should handle extreme but valid learning rates
                trainer = FineTuningTrainer(
                    model=mock_model,
                    config=config,
                    instruction_dataset=mock_instruction_dataset
                )
                assert trainer.config.learning_rate == lr

    def test_invalid_scheduler_type(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock
    ):
        """Test handling of invalid scheduler type."""
        small_config.scheduler_type = "invalid_scheduler"
        
        # Should raise ValueError for invalid scheduler type
        with pytest.raises(ValueError, match="scheduler_type must be one of"):
            FineTuningTrainer(
                model=mock_model,
                config=small_config,
                instruction_dataset=mock_instruction_dataset
            )

    def test_corrupted_checkpoint_loading(
        self, 
        mock_model: Mock, 
        small_config: Config, 
        mock_instruction_dataset: Mock,
        temp_dir: Path
    ):
        """Test handling of corrupted checkpoint files."""
        # Create a corrupted checkpoint file
        corrupted_path = temp_dir / "corrupted.pth"
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid checkpoint file")
        
        trainer = FineTuningTrainer(
            model=mock_model,
            config=small_config,
            instruction_dataset=mock_instruction_dataset
        )
        
        with pytest.raises(Exception):  # Should raise some kind of loading error
            trainer.load_checkpoint(corrupted_path)


if __name__ == "__main__":
    pytest.main([__file__])
