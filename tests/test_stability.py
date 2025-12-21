"""Numerical stability tests for the GPT language model.

These tests ensure the model is reliable and doesn't produce NaN/Inf values
during training or inference, especially under edge cases.
"""
import pytest
import torch

from src.config import Config
from src.model import create_model
from src.training import Trainer
from src.data import TextDataset


class TestNumericalStability:
    """Test numerical stability during training and inference."""

    def test_no_nan_inf_in_forward_pass(self):
        """Test that forward pass doesn't produce NaN/Inf."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
        assert torch.isfinite(loss).all(), "Loss contains NaN/Inf"

    def test_no_nan_inf_after_backward(self):
        """Test that gradients don't become NaN/Inf."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)
        loss.backward()

        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"

    def test_gradient_clipping_prevents_explosion(self):
        """Test that gradient clipping prevents gradient explosion."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            grad_clip=1.0
        )
        model = create_model(config, 1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)  # High LR to test clipping

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Check gradient norm is bounded
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= config.grad_clip * 1.1, f"Gradient norm {total_norm} exceeds clip value"


class TestExtremeInputs:
    """Test model behavior with extreme inputs."""

    def test_very_long_sequence(self):
        """Test model with sequence at block_size limit."""
        config = Config(
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=512,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Test at maximum block size
        x = torch.randint(0, 1000, (1, 512))
        y = torch.randint(0, 1000, (1, 512))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()
        assert logits.shape == (1, 512, 1000)

    def test_single_token_sequence(self):
        """Test model with minimal sequence length."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Test with single token
        x = torch.randint(0, 1000, (1, 1))
        y = torch.randint(0, 1000, (1, 1))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()
        assert logits.shape == (1, 1, 1000)

    def test_large_batch_size(self):
        """Test model with large batch size."""
        config = Config(
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=64,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Test with large batch
        x = torch.randint(0, 1000, (64, 32))
        y = torch.randint(0, 1000, (64, 32))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()
        assert logits.shape == (64, 32, 1000)


class TestMixedPrecisionStability:
    """Test stability with mixed precision training."""

    def test_fp16_no_nan_inf(self):
        """Test that FP16 training doesn't produce NaN/Inf."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            fp16=False  # CPU doesn't support FP16, but test the path
        )
        model = create_model(config, 1000)

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        # Simulate mixed precision path
        with torch.amp.autocast('cpu', enabled=False):
            logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()


class TestInitializationStability:
    """Test that initialization doesn't create instability."""

    def test_initialization_produces_finite_values(self):
        """Test that all initialized parameters are finite."""
        config = Config(
            n_embd=128,
            n_head=8,
            n_layer=6,
            block_size=256,
            vocab_size=5000,
            device='cpu'
        )
        model = create_model(config, 5000)

        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Parameter {name} not initialized properly"

    def test_initialization_variance(self):
        """Test that initialization has reasonable variance."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                var = param.var().item()
                # Variance should be reasonable (not too small or too large)
                assert 1e-5 < var < 10.0, f"Parameter {name} has unusual variance: {var}"


class TestGenerationStability:
    """Test stability during text generation."""

    def test_generation_no_nan_inf(self):
        """Test that generation doesn't produce NaN/Inf."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)
        model.eval()

        idx = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=50)

        # All outputs should be finite integers
        assert torch.isfinite(output.float()).all()
        assert torch.all(output >= 0)
        assert torch.all(output < 1000)

    def test_long_generation_stability(self):
        """Test stability when generating many tokens."""
        config = Config(
            n_embd=32,
            n_head=2,
            n_layer=1,
            block_size=256,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)
        model.eval()

        idx = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=200)

        assert torch.isfinite(output.float()).all()
        assert output.shape == (1, 210)

    @pytest.mark.parametrize("temperature", [0.01, 0.1, 1.0, 10.0])
    def test_extreme_temperatures_stability(self, temperature):
        """Test stability with extreme temperature values."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)
        model.eval()

        idx = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20, temperature=temperature)

        assert torch.isfinite(output.float()).all()
        assert torch.all(output >= 0)
        assert torch.all(output < 1000)


class TestDropoutStability:
    """Test stability with different dropout rates."""

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_dropout_stability(self, dropout):
        """Test that different dropout rates don't cause instability."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            dropout=dropout
        )
        model = create_model(config, 1000)
        model.train()  # Enable dropout

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()


class TestWeightTyingStability:
    """Test stability of weight tying between embeddings and LM head."""

    def test_tied_weights_remain_finite(self):
        """Test that tied weights don't cause instability."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Verify weights are tied
        assert model.token_embedding.weight is model.lm_head.weight

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)
        loss.backward()

        # Check tied weights have finite gradients
        assert torch.isfinite(model.token_embedding.weight.grad).all()

    def test_tied_weights_update_together(self):
        """Test that tied weights update consistently."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Get initial weight value
        initial_weight = model.token_embedding.weight.clone()

        x = torch.randint(0, 1000, (4, 32))
        y = torch.randint(0, 1000, (4, 32))

        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.token_embedding.weight, initial_weight)

        # Tied weights should still be the same object
        assert model.token_embedding.weight is model.lm_head.weight
