"""Tests for recursive reasoning GPT model.

This module tests the RecursiveGPTLanguageModel and its components.
"""

import pytest
import torch

from src.config import Config
from src.model.recursive import (
    RecursiveGPTLanguageModel,
    RecursiveTransformerBlock,
    ConfidenceHead,
    create_recursive_model
)


class TestConfidenceHead:
    """Test the confidence head component."""

    def test_forward_shape(self):
        """Test that confidence head produces correct output shape."""
        head = ConfidenceHead(n_embd=64)
        x = torch.randn(2, 10, 64)

        confidence = head(x)

        assert confidence.shape == (2, 10, 1)

    def test_different_embeddings(self):
        """Test confidence head with different embedding dimensions."""
        for n_embd in [32, 64, 128, 256]:
            head = ConfidenceHead(n_embd=n_embd)
            x = torch.randn(2, 10, n_embd)

            confidence = head(x)

            assert confidence.shape == (2, 10, 1)


class TestRecursiveTransformerBlock:
    """Test the recursive transformer block."""

    def test_forward_without_states(self):
        """Test forward pass without provided states."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )

        block = RecursiveTransformerBlock(config)
        x = torch.randn(2, 10, 64)

        output, y, z, lb_loss = block(x)

        assert output.shape == (2, 10, 64)
        assert y.shape == (2, 10, 64)
        assert z.shape == (2, 10, 64)

    def test_forward_with_states(self):
        """Test forward pass with provided states."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )

        block = RecursiveTransformerBlock(config)
        x = torch.randn(2, 10, 64)
        y_prev = torch.randn(2, 10, 64)
        z_prev = torch.randn(2, 10, 64)

        output, y, z, lb_loss = block(x, y_prev, z_prev)

        assert output.shape == (2, 10, 64)
        assert y.shape == (2, 10, 64)
        assert z.shape == (2, 10, 64)
        # States should have changed
        assert not torch.allclose(y, y_prev)
        assert not torch.allclose(z, z_prev)


class TestRecursiveGPTLanguageModel:
    """Test the recursive GPT language model."""

    def test_recursive_disabled(self):
        """Test model with recursion disabled."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=False
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))

        logits, loss = model(x)

        assert logits.shape == (2, 32, 1000)
        assert loss is None

    def test_recursive_enabled(self):
        """Test model with recursion enabled."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=2,
            latent_steps=3
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))

        logits, loss = model(x)

        assert logits.shape == (2, 32, 1000)
        assert loss is None

    def test_recursive_with_targets(self):
        """Test recursive model with target tokens."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=2,
            latent_steps=3
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))

        logits, loss = model(x, y)

        assert logits.shape == (2, 32, 1000)
        assert loss is not None
        assert loss.ndim == 0  # Scalar loss
        assert torch.isfinite(loss)

    def test_recursive_with_confidence(self):
        """Test recursive model returning confidence scores."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=2,
            latent_steps=3
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))

        logits, loss, confidence = model(x, return_confidence=True)

        assert logits.shape == (2, 32, 1000)
        assert confidence.shape == (2, 32, 1)
        # Confidence should be between 0 and 1 (sigmoid output)
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)

    def test_generation(self):
        """Test text generation with recursive model."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=2,
            latent_steps=3
        )

        model = RecursiveGPTLanguageModel(config)
        model.eval()

        idx = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20)

        assert output.shape == (1, 30)  # 10 + 20
        assert torch.all(output >= 0)
        assert torch.all(output < 1000)

    def test_generation_without_recursion(self):
        """Test that we can disable recursion during generation."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=2,
            latent_steps=3
        )

        model = RecursiveGPTLanguageModel(config)
        model.eval()

        idx = torch.randint(0, 1000, (1, 10))

        # Generate without recursion (faster)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20, use_recursion=False)

        assert output.shape == (1, 30)

    def test_parameter_count_smaller_than_baseline(self):
        """Test that recursive model can be smaller than equivalent baseline."""
        # Baseline model: 4 layers, no recursion
        baseline_config = Config(
            n_embd=64,
            n_head=4,
            n_layer=4,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=False
        )
        baseline_model = RecursiveGPTLanguageModel(baseline_config)
        baseline_params = sum(p.numel() for p in baseline_model.parameters())

        # Recursive model: 2 layers + recursion (should be smaller)
        recursive_config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=3,
            latent_steps=4
        )
        recursive_model = RecursiveGPTLanguageModel(recursive_config)
        recursive_params = sum(p.numel() for p in recursive_model.parameters())

        # Recursive model should have fewer parameters
        # (trading parameters for computation)
        assert recursive_params < baseline_params

        print(f"\nBaseline: {baseline_params:,} params")
        print(f"Recursive: {recursive_params:,} params")
        print(f"Savings: {(1 - recursive_params/baseline_params)*100:.1f}%")


class TestCreateRecursiveModel:
    """Test the factory function."""

    def test_create_recursive_model(self):
        """Test creating model with factory function."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            device='cpu',
            use_recursive=True
        )

        model = create_recursive_model(config, vocab_size=1000)

        assert isinstance(model, RecursiveGPTLanguageModel)
        assert model.config.vocab_size == 1000

    def test_backward_compatibility(self):
        """Test that model works without recursive parameters."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
            # No recursive parameters - should default to disabled
        )

        model = RecursiveGPTLanguageModel(config)

        # Should work without errors
        x = torch.randint(0, 1000, (2, 32))
        logits, loss = model(x)

        assert logits.shape == (2, 32, 1000)


class TestNumericalStability:
    """Test numerical stability of recursive model."""

    def test_no_nan_inf_with_recursion(self):
        """Test that recursion doesn't cause NaN/Inf."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=5,  # Deep recursion
            latent_steps=10  # Many latent steps
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()

    def test_gradients_finite_with_recursion(self):
        """Test that gradients are finite with recursion."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            use_recursive=True,
            recursion_depth=3,
            latent_steps=5
        )

        model = RecursiveGPTLanguageModel(config)
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))

        logits, loss = model(x, y)
        loss.backward()

        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"
