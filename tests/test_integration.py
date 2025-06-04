"""Integration tests for the complete pipeline."""
import pytest
import torch

from src.config import Config
from src.model import GPTLanguageModel
from src.tokenizer import BPETokenizer
from src.data import TextDataset
from src.training import Trainer
from src.config import setup_logging


class TestIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.mark.integration
    def test_complete_training_pipeline(self, temp_dir, sample_text_data, vocab_files):
        """Test the complete training pipeline from start to finish."""
        # Setup logging
        setup_logging(level="INFO")

        # Create configuration
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=2,
            eval_interval=1,
            save_interval=1,
            batch_size=2,
            learning_rate=1e-3,
            device="cpu"
        )

        # Initialize tokenizer
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Create datasets
        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        val_dataset = TextDataset(
            text=sample_text_data[:len(sample_text_data) // 2],
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        # Initialize model
        model = GPTLanguageModel(config)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.current_epoch == config.max_epochs
        assert trainer.global_step > 0

        # Verify checkpoints were saved
        checkpoint_files = list(temp_dir.glob("checkpoint_*.pth"))
        assert len(checkpoint_files) > 0

        # Test text generation
        generated_text = trainer.generate_text(
            prompt="hello",
            tokenizer=tokenizer,
            max_tokens=5,
            temperature=1.0
        )

        assert isinstance(generated_text, str)
        assert len(generated_text) > 5

    @pytest.mark.integration
    def test_model_save_and_load(self, temp_dir, sample_text_data, vocab_files):
        """Test saving a trained model and loading it in a new session."""
        # Create and train a small model
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=1,
            batch_size=2,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        # Train original model
        original_model = GPTLanguageModel(config)
        original_trainer = Trainer(
            model=original_model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Train for a few steps
        original_trainer.train_epoch()

        # Save checkpoint
        checkpoint_path = original_trainer.save_checkpoint()

        # Create new model and trainer
        new_model = GPTLanguageModel(config)
        new_trainer = Trainer(
            model=new_model,
            config=config,
            train_dataset=None,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)

        # Compare model outputs
        original_model.eval()
        new_model.eval()

        test_input = torch.randint(0, config.vocab_size, (1, 4))

        with torch.no_grad():
            original_output, _ = original_model(test_input)
            new_output, _ = new_model(test_input)

        # Outputs should be identical
        assert torch.allclose(original_output, new_output, atol=1e-6)

        # Training state should be preserved
        assert new_trainer.current_epoch == original_trainer.current_epoch
        assert new_trainer.global_step == original_trainer.global_step

    @pytest.mark.integration
    def test_tokenizer_model_compatibility(self, temp_dir, vocab_files):
        """Test that tokenizer and model work together correctly."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        config = Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            device="cpu"
        )

        model = GPTLanguageModel(config)

        # Test text encoding and model forward pass
        text = "hello world test"
        tokens = tokenizer.encode(text)

        if len(tokens) > 0:
            # Prepare input for model
            input_tensor = torch.tensor(tokens[:config.block_size]).unsqueeze(0)

            # Forward pass
            logits, _ = model(input_tensor)

            # Check output shape
            expected_shape = (1, input_tensor.shape[1], config.vocab_size)
            assert logits.shape == expected_shape

            # Test generation
            model.eval()
            with torch.no_grad():
                generated_tokens = model.generate(input_tensor, max_new_tokens=3)

            # Decode generated tokens
            generated_text = tokenizer.decode(generated_tokens[0].tolist())
            assert isinstance(generated_text, str)

    @pytest.mark.integration
    def test_different_configurations(self, temp_dir, sample_text_data, vocab_files):
        """Test training with different model configurations."""
        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        configs = [
            # Small config
            Config(
                vocab_size=30,
                n_embd=16,
                n_head=2,
                n_layer=1,
                block_size=4,
                max_epochs=1,
                batch_size=1,
                device="cpu"
            ),
            # Medium config
            Config(
                vocab_size=50,
                n_embd=32,
                n_head=4,
                n_layer=2,
                block_size=8,
                max_epochs=1,
                batch_size=2,
                device="cpu"
            ),
        ]

        for i, config in enumerate(configs):
            # Create model and dataset
            model = GPTLanguageModel(config)
            dataset = TextDataset(
                text=sample_text_data,
                tokenizer=tokenizer,
                block_size=config.block_size
            )

            # Create trainer with subdirectory
            config_dir = temp_dir / f"config_{i}"
            trainer = Trainer(
                model=model,
                config=config,
                train_dataset=dataset,
                val_dataset=None,
                save_dir=config_dir
            )

            # Run training
            trainer.train()

            # Verify training completed
            assert trainer.current_epoch == config.max_epochs

            # Test text generation
            generated_text = trainer.generate_text(
                prompt="test",
                tokenizer=tokenizer,
                max_tokens=3,
                temperature=1.0
            )
            assert isinstance(generated_text, str)

    @pytest.mark.integration
    def test_training_resumption(self, temp_dir, sample_text_data, vocab_files):
        """Test resuming training from a checkpoint."""
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=4,  # Will train in two sessions
            eval_interval=1,
            save_interval=1,
            batch_size=2,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        # First training session - train for 2 epochs
        model1 = GPTLanguageModel(config)
        trainer1 = Trainer(
            model=model1,
            config=config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Manually train for 2 epochs
        for _ in range(2):
            trainer1.train_epoch()
            trainer1.save_checkpoint()

        checkpoint_files = list(temp_dir.glob("checkpoint_*.pth"))
        assert len(checkpoint_files) >= 1

        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        # Second training session - resume and complete training
        model2 = GPTLanguageModel(config)
        trainer2 = Trainer(
            model=model2,
            config=config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Load checkpoint
        trainer2.load_checkpoint(latest_checkpoint)

        # Verify state was loaded
        assert trainer2.current_epoch == 2

        # Complete training
        trainer2.train()

        # Should have trained for the remaining epochs
        assert trainer2.current_epoch == config.max_epochs

    @pytest.mark.integration
    def test_evaluation_during_training(self, temp_dir, sample_text_data, vocab_files):
        """Test that evaluation works correctly during training."""
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=3,
            eval_interval=1,  # Evaluate every epoch
            batch_size=2,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Create train and validation datasets
        full_text = sample_text_data
        split_point = len(full_text) // 2

        train_dataset = TextDataset(
            text=full_text[:split_point],
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        val_dataset = TextDataset(
            text=full_text[split_point:],
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        model = GPTLanguageModel(config)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        # Run training
        trainer.train()

        # Should have completed training
        assert trainer.current_epoch == config.max_epochs

        # Should have recorded validation losses
        assert trainer.best_val_loss < float('inf')

    @pytest.mark.integration
    @pytest.mark.slow
    def test_larger_model_training(self, temp_dir, sample_text_data, vocab_files):
        """Test training with a larger model configuration."""
        config = Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=3,
            block_size=16,
            max_epochs=1,
            eval_interval=1,
            save_interval=1,
            batch_size=4,
            learning_rate=5e-4,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Create larger dataset by repeating text
        large_text = sample_text_data * 5

        train_dataset = TextDataset(
            text=large_text,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        val_dataset = TextDataset(
            text=sample_text_data,
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        model = GPTLanguageModel(config)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=temp_dir
        )

        # Check model size
        from src.utils.helpers import count_parameters
        total_params, trainable_params = count_parameters(model)
        assert total_params > 10000  # Should be reasonably large
        assert trainable_params == total_params

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.current_epoch == config.max_epochs

        # Test generation with different temperatures
        for temperature in [0.5, 1.0, 1.5]:
            generated_text = trainer.generate_text(
                prompt="test",
                tokenizer=tokenizer,
                max_tokens=10,
                temperature=temperature
            )
            assert isinstance(generated_text, str)
            assert len(generated_text) > 4  # Should be longer than prompt

    @pytest.mark.integration
    def test_error_recovery(self, temp_dir, sample_text_data, vocab_files):
        """Test system behavior with various error conditions."""
        config = Config(
            vocab_size=50,
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            max_epochs=1,
            batch_size=2,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        # Test with very small dataset
        tiny_dataset = TextDataset(
            text="a",  # Very short text
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        model = GPTLanguageModel(config)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=tiny_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Should handle small dataset gracefully
        if len(tiny_dataset) > 0:
            trainer.train()

        # Test with mismatched vocab sizes
        config_mismatch = Config(
            vocab_size=1000,  # Much larger than tokenizer vocab
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=8,
            device="cpu"
        )

        large_vocab_model = GPTLanguageModel(config_mismatch)

        # Should create model without error, but may have issues during training
        assert large_vocab_model.token_embedding.num_embeddings == 1000

    @pytest.mark.integration
    def test_memory_efficiency(self, temp_dir, sample_text_data, vocab_files):
        """Test that the system doesn't use excessive memory."""
        import gc
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        config = Config(
            vocab_size=100,
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=16,
            max_epochs=1,
            batch_size=4,
            device="cpu"
        )

        tokenizer = BPETokenizer(
            encoder_file=vocab_files["encoder"],
            decoder_file=vocab_files["decoder"]
        )

        train_dataset = TextDataset(
            text=sample_text_data * 3,  # Larger dataset
            tokenizer=tokenizer,
            block_size=config.block_size
        )

        model = GPTLanguageModel(config)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=None,
            save_dir=temp_dir
        )

        # Run training
        trainer.train()

        # Clean up
        del trainer, model, train_dataset
        gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this small model)
        assert memory_increase < 500, f"Memory usage increased by {
            memory_increase:.1f}MB"
