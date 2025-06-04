"""Tests for the trainer module."""
import pytest
import torch

from src.training import Trainer
from src.model import GPTLanguageModel
from src.data import TextDataset
from src.tokenizer import BPETokenizer
from src.config import Config


class TestTrainer:
    """Test cases for Trainer class."""

    def test_trainer_initialization(self, small_model_config, temp_dir):
        """Test Trainer initialization."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        assert trainer.model is model
        assert trainer.config is small_model_config
        assert trainer.save_dir == temp_dir
        assert trainer.device == small_model_config.get_device()
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')

        # Check optimizer initialization
        assert trainer.optimizer is not None
        assert trainer.optimizer.__class__.__name__ == "AdamW"

    def test_trainer_with_datasets(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test Trainer initialization with datasets."""
        model = GPTLanguageModel(small_model_config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        val_dataset = TextDataset(
            text=sample_text_data[:len(sample_text_data) // 2],  # Smaller val set
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        assert trainer.train_dataset is train_dataset
        assert trainer.val_dataset is val_dataset
        assert trainer.train_dataloader is not None
        assert trainer.val_dataloader is not None

    def test_trainer_step(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test single training step."""
        model = GPTLanguageModel(small_model_config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Get a batch
        x, y = train_dataset[0]
        x = x.unsqueeze(0)  # Add batch dimension
        y = y.unsqueeze(0)

        initial_step = trainer.global_step
        loss = trainer.step(x, y)

        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive
        assert trainer.global_step == initial_step + 1

    def test_trainer_evaluate(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test model evaluation."""
        model = GPTLanguageModel(small_model_config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        val_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        avg_loss = trainer.evaluate()

        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_trainer_evaluate_no_dataset(self, small_model_config, temp_dir):
        """Test evaluation without validation dataset."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        avg_loss = trainer.evaluate()
        assert avg_loss is None

    def test_trainer_save_checkpoint(self, small_model_config, temp_dir):
        """Test saving training checkpoint."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Set some training state
        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 2.5

        checkpoint_path = trainer.save_checkpoint()

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pth"

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["best_val_loss"] == 2.5

    def test_trainer_load_checkpoint(self, small_model_config, temp_dir):
        """Test loading training checkpoint."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Save a checkpoint first
        trainer.current_epoch = 3
        trainer.global_step = 75
        trainer.best_val_loss = 1.8
        checkpoint_path = trainer.save_checkpoint()

        # Create new trainer and load checkpoint
        new_model = GPTLanguageModel(small_model_config)
        new_trainer = Trainer(
            model=new_model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.current_epoch == 3
        assert new_trainer.global_step == 75
        assert new_trainer.best_val_loss == 1.8

    def test_trainer_load_nonexistent_checkpoint(self, small_model_config, temp_dir):
        """Test loading non-existent checkpoint."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("nonexistent.pth")

    def test_trainer_generate_text(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test text generation."""
        model = GPTLanguageModel(small_model_config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        prompt = "hello"
        generated_text = trainer.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=10,
            temperature=1.0
        )

        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
        assert generated_text.startswith(prompt)

    def test_trainer_generate_text_no_tokenizer(self, small_model_config, temp_dir):
        """Test text generation without tokenizer raises error."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        with pytest.raises(ValueError, match="Tokenizer is required"):
            trainer.generate_text("hello", None, max_tokens=10)

    def test_trainer_train_epoch(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test training for one epoch."""
        model = GPTLanguageModel(small_model_config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        initial_epoch = trainer.current_epoch
        initial_step = trainer.global_step

        avg_loss = trainer.train_epoch()

        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert trainer.current_epoch == initial_epoch + 1
        assert trainer.global_step > initial_step

    def test_trainer_train_epoch_no_dataset(self, small_model_config, temp_dir):
        """Test training epoch without training dataset."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        with pytest.raises(RuntimeError, match="No training dataset"):
            trainer.train_epoch()

    def test_trainer_full_training_loop(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test full training loop with very short training."""
        # Use very small config for fast test
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=2,  # Very short training
            eval_interval=1,
            save_interval=1,
            batch_size=2,
            device="cpu"
        )

        model = GPTLanguageModel(config)
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        val_dataset = TextDataset(
            text=sample_text_data[:50],  # Very small val set
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        # Run training
        trainer.train()

        assert trainer.current_epoch == config.max_epochs
        assert trainer.global_step > 0

        # Check that checkpoints were saved
        checkpoint_files = list(temp_dir.glob("checkpoint_*.pth"))
        assert len(checkpoint_files) > 0

    @pytest.mark.slow
    def test_trainer_mixed_precision(
            self,
            small_model_config,
            sample_text_data,
            vocab_files,
            temp_dir):
        """Test mixed precision training if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        # Update config for CUDA
        small_model_config.device = "cuda"

        model = GPTLanguageModel(small_model_config).to("cuda")
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=small_model_config.block_size
        )

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir,
            use_mixed_precision=True
        )

        # Check that scaler is initialized
        assert trainer.scaler is not None

        # Run one training step
        x, y = train_dataset[0]
        x = x.unsqueeze(0).to("cuda")
        y = y.unsqueeze(0).to("cuda")

        loss = trainer.step(x, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_trainer_device_handling(self, small_model_config, temp_dir):
        """Test proper device handling."""
        device = "cpu"
        small_model_config.device = device

        model = GPTLanguageModel(small_model_config).to(device)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        assert trainer.device == device

        # Check that model is on correct device
        for param in trainer.model.parameters():
            assert param.device.type == device

    def test_trainer_optimizer_configuration(self, small_model_config, temp_dir):
        """Test optimizer configuration."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Check optimizer type and learning rate
        assert trainer.optimizer.__class__.__name__ == "AdamW"
        assert trainer.optimizer.param_groups[0]['lr'] == small_model_config.learning_rate

        # Check that all model parameters are being optimized
        optimizer_params = set()
        for group in trainer.optimizer.param_groups:
            for param in group['params']:
                optimizer_params.add(id(param))

        model_params = set(id(param)
                           for param in model.parameters() if param.requires_grad)
        assert optimizer_params == model_params

    def test_trainer_logging(self, small_model_config, temp_dir, caplog):
        """Test that trainer produces appropriate log messages."""
        import logging

        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Test logging during checkpoint save
        with caplog.at_level(logging.INFO):
            checkpoint_path = trainer.save_checkpoint()

        # Should have logged about saving checkpoint
        assert any("Saved checkpoint" in record.message for record in caplog.records)
        # Verify checkpoint was actually created
        assert checkpoint_path.exists()

    def test_trainer_state_consistency(self, small_model_config, temp_dir):
        """Test that trainer state remains consistent."""
        model = GPTLanguageModel(small_model_config)

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Initial state
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')

        # After manually updating state
        trainer.current_epoch = 10
        trainer.global_step = 1000
        trainer.best_val_loss = 1.5

        # Save and reload
        checkpoint_path = trainer.save_checkpoint()

        new_trainer = Trainer(
            model=GPTLanguageModel(small_model_config),
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.current_epoch == 10
        assert new_trainer.global_step == 1000
        assert new_trainer.best_val_loss == 1.5

    def test_trainer_error_handling(self, small_model_config, temp_dir):
        """Test error handling in trainer."""
        model = GPTLanguageModel(small_model_config)

        # Test with invalid save directory
        invalid_dir = temp_dir / "nonexistent" / "deeply" / "nested"

        trainer = Trainer(
            model=model,
            config=small_model_config,
            train_dataset=None,
            val_dataset=None,
            save_dir=invalid_dir
        )

        # Should create directory and save successfully
        checkpoint_path = trainer.save_checkpoint()
        assert checkpoint_path.exists()
        assert invalid_dir.exists()
