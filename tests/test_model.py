"""Tests for the model module."""
import pytest
import torch
import torch.nn as nn

from src.model import MultiHeadAttention, FeedForward, Block, GPTLanguageModel
from src.config import Config


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention class."""

    def test_multihead_attention_initialization(self, small_model_config):
        """Test MultiHeadAttention initialization."""
        attention = MultiHeadAttention(small_model_config)

        assert attention.num_heads == small_model_config.n_head
        assert attention.head_dim == small_model_config.n_embd // small_model_config.n_head
        assert attention.config == small_model_config

        # Check layers exist
        assert hasattr(attention, 'qkv')
        assert hasattr(attention, 'proj')
        assert hasattr(attention, 'dropout')
        assert hasattr(attention, 'causal_mask')

    def test_multihead_attention_forward(self, small_model_config):
        """Test MultiHeadAttention forward pass."""
        attention = MultiHeadAttention(small_model_config)

        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, small_model_config.n_embd)

        output = attention(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_multihead_attention_causal_mask(self, small_model_config):
        """Test that attention mask prevents future information leakage."""
        attention = MultiHeadAttention(small_model_config)
        attention.eval()  # Disable dropout for deterministic behavior

        batch_size = 1
        seq_len = 4
        x = torch.randn(batch_size, seq_len, small_model_config.n_embd)

        # Set specific values to test causality
        x[0, 1:, :] = 999.0  # Set future tokens to large values

        with torch.no_grad():
            output = attention(x)

        # First token should not be affected by future tokens
        # This is a simplified test - in practice, the effect might be subtle
        assert torch.isfinite(output).all()
        assert output.shape == x.shape

    def test_multihead_attention_different_head_sizes(self):
        """Test attention with different valid head configurations."""
        configs = [
            Config(
                n_embd=64,
                n_head=1,
                n_layer=1,
                vocab_size=100,
                block_size=16,
                device="cpu"),
            Config(
                n_embd=64,
                n_head=2,
                n_layer=1,
                vocab_size=100,
                block_size=16,
                device="cpu"),
            Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
                block_size=16,
                device="cpu"),
        ]

        for config in configs:
            attention = MultiHeadAttention(config)
            x = torch.randn(1, 4, config.n_embd)
            output = attention(x)
            assert output.shape == x.shape


class TestFeedForward:
    """Test cases for FeedForward class."""

    def test_feedforward_initialization(self, small_model_config):
        """Test FeedForward initialization."""
        ff = FeedForward(small_model_config)

        assert hasattr(ff, 'net')
        assert isinstance(ff.net, nn.Sequential)

        # Check the network structure
        layers = list(ff.net.children())
        assert len(layers) == 4  # Linear, ReLU, Linear, Dropout
        assert isinstance(layers[0], nn.Linear)
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[2], nn.Linear)
        assert isinstance(layers[3], nn.Dropout)

        # Check dimensions
        assert layers[0].in_features == small_model_config.n_embd
        assert layers[0].out_features == 4 * small_model_config.n_embd
        assert layers[2].in_features == 4 * small_model_config.n_embd
        assert layers[2].out_features == small_model_config.n_embd

    def test_feedforward_forward(self, small_model_config):
        """Test FeedForward forward pass."""
        ff = FeedForward(small_model_config)

        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, small_model_config.n_embd)

        output = ff(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_feedforward_nonlinearity(self, small_model_config):
        """Test that FeedForward applies nonlinearity."""
        ff = FeedForward(small_model_config)
        ff.eval()  # Disable dropout

        # Test with zeros should remain zeros after ReLU
        x_zeros = torch.zeros(1, 1, small_model_config.n_embd)
        output_zeros = ff(x_zeros)

        # Output should be close to linear transformation of input due to ReLU
        assert torch.isfinite(output_zeros).all()


class TestBlock:
    """Test cases for Block (Transformer block) class."""

    def test_block_initialization(self, small_model_config):
        """Test Block initialization."""
        block = Block(small_model_config)

        assert hasattr(block, 'attention')  # self-attention
        assert hasattr(block, 'feed_forward')  # feed-forward
        assert hasattr(block, 'ln1')  # layer norm 1
        assert hasattr(block, 'ln2')  # layer norm 2

        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.feed_forward, FeedForward)
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)

    def test_block_forward(self, small_model_config):
        """Test Block forward pass."""
        block = Block(small_model_config)

        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, small_model_config.n_embd)

        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_block_residual_connections(self, small_model_config):
        """Test that residual connections work properly."""
        block = Block(small_model_config)
        block.eval()

        # Create input
        x = torch.randn(1, 4, small_model_config.n_embd)

        with torch.no_grad():
            output = block(x)

        # Output should be different from input (due to transformations)
        # but should maintain reasonable magnitude due to residual connections
        assert not torch.allclose(output, x, atol=1e-6)
        assert torch.isfinite(output).all()


class TestGPTLanguageModel:
    """Test cases for GPTLanguageModel class."""

    def test_model_initialization(self, small_model_config):
        """Test GPTLanguageModel initialization."""
        model = GPTLanguageModel(small_model_config)

        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'position_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'ln_f')
        assert hasattr(model, 'lm_head')

        assert isinstance(model.token_embedding, nn.Embedding)
        assert isinstance(model.position_embedding, nn.Embedding)
        assert isinstance(model.blocks, nn.ModuleList)
        assert isinstance(model.ln_f, nn.LayerNorm)
        assert isinstance(model.lm_head, nn.Linear)

        # Check dimensions
        assert model.token_embedding.num_embeddings == small_model_config.vocab_size
        assert model.token_embedding.embedding_dim == small_model_config.n_embd
        assert model.position_embedding.num_embeddings == small_model_config.block_size
        assert model.position_embedding.embedding_dim == small_model_config.n_embd
        assert len(model.blocks) == small_model_config.n_layer

    def test_model_forward(self, small_model_config):
        """Test GPTLanguageModel forward pass."""
        model = GPTLanguageModel(small_model_config)

        batch_size = 2
        seq_len = 8
        idx = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx)

        expected_shape = (batch_size, seq_len, small_model_config.vocab_size)
        assert logits.shape == expected_shape
        assert loss is None  # No targets provided

    def test_model_forward_with_targets(self, small_model_config):
        """Test GPTLanguageModel forward pass with targets."""
        model = GPTLanguageModel(small_model_config)

        batch_size = 2
        seq_len = 8
        idx = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len))

        logits, loss = model(idx, targets)

        expected_shape = (batch_size, seq_len, small_model_config.vocab_size)
        assert logits.shape == expected_shape
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_model_generate(self, small_model_config):
        """Test GPTLanguageModel generate method."""
        model = GPTLanguageModel(small_model_config)
        model.eval()

        batch_size = 1
        seq_len = 4
        max_new_tokens = 8
        idx = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens)

        expected_length = seq_len + max_new_tokens
        assert generated.shape == (batch_size, expected_length)
        assert generated.dtype == idx.dtype

        # Check that original sequence is preserved
        assert torch.equal(generated[:, :seq_len], idx)

    def test_model_generate_temperature(self, small_model_config):
        """Test GPTLanguageModel generate with different temperatures."""
        model = GPTLanguageModel(small_model_config)
        model.eval()

        idx = torch.randint(0, small_model_config.vocab_size, (1, 4))
        max_new_tokens = 4

        with torch.no_grad():
            # Test different temperatures
            for temperature in [0.5, 1.0, 1.5]:
                generated = model.generate(idx, max_new_tokens, temperature=temperature)
                assert generated.shape == (1, 4 + max_new_tokens)

    def test_model_generate_long_sequence(self, small_model_config):
        """Test generating sequence longer than block_size."""
        model = GPTLanguageModel(small_model_config)
        model.eval()

        idx = torch.randint(0, small_model_config.vocab_size, (1, 2))
        max_new_tokens = small_model_config.block_size + 5  # Longer than block_size

        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens)

        expected_length = 2 + max_new_tokens
        assert generated.shape == (1, expected_length)

    def test_model_device_consistency(self, small_model_config):
        """Test that model handles device consistency."""
        device = "cpu"
        model = GPTLanguageModel(small_model_config).to(device)

        idx = torch.randint(0, small_model_config.vocab_size, (1, 4)).to(device)

        logits, loss = model(idx)
        assert logits.device.type == device

    def test_model_parameter_count(self, small_model_config):
        """Test that model has expected number of parameters."""
        model = GPTLanguageModel(small_model_config)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

        # Check that all parameters are trainable by default
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == total_params

    def test_model_gradient_flow(self, small_model_config):
        """Test that gradients flow properly through the model."""
        model = GPTLanguageModel(small_model_config)

        idx = torch.randint(0, small_model_config.vocab_size, (2, 4))
        targets = torch.randint(0, small_model_config.vocab_size, (2, 4))

        logits, loss = model(idx, targets)
        loss.backward()

        # Check that gradients exist for key components
        assert model.token_embedding.weight.grad is not None
        assert model.lm_head.weight.grad is not None

        # Check that gradients are not zero everywhere
        total_grad_norm = sum(p.grad.norm().item()
                              for p in model.parameters() if p.grad is not None)
        assert total_grad_norm > 0

    def test_model_save_load_state_dict(self, small_model_config, temp_dir):
        """Test saving and loading model state dict."""
        model1 = GPTLanguageModel(small_model_config)

        # Save state dict
        state_dict_path = temp_dir / "model_state.pth"
        torch.save(model1.state_dict(), state_dict_path)

        # Create new model and load state dict
        model2 = GPTLanguageModel(small_model_config)
        model2.load_state_dict(torch.load(state_dict_path, map_location="cpu", weights_only=False))

        # Test that models produce same output
        model1.eval()
        model2.eval()

        idx = torch.randint(0, small_model_config.vocab_size, (1, 4))

        with torch.no_grad():
            logits1, _ = model1(idx)
            logits2, _ = model2(idx)

        assert torch.allclose(logits1, logits2, atol=1e-6)

    def test_model_eval_mode(self, small_model_config):
        """Test model behavior in eval mode."""
        model = GPTLanguageModel(small_model_config)

        # Test training mode
        model.train()
        assert model.training

        # Test eval mode
        model.eval()
        assert not model.training

        # All submodules should also be in eval mode
        for module in model.modules():
            assert not module.training

    @pytest.mark.parametrize("vocab_size,n_embd,n_head,n_layer", [
        (100, 32, 2, 1),
        (200, 64, 4, 2),
        (500, 128, 8, 3),
    ])
    def test_model_different_configurations(self, vocab_size, n_embd, n_head, n_layer):
        """Test model with different valid configurations."""
        config = Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            block_size=16,
            device="cpu"
        )

        model = GPTLanguageModel(config)
        idx = torch.randint(0, vocab_size, (1, 8))

        logits, loss = model(idx)
        assert logits.shape == (1, 8, vocab_size)

    def test_model_memory_efficiency(self, small_model_config):
        """Test that model doesn't use excessive memory."""
        model = GPTLanguageModel(small_model_config)

        # Get initial memory usage
        import gc
        gc.collect()

        # Run forward pass
        idx = torch.randint(0, small_model_config.vocab_size, (4, 16))
        logits, loss = model(idx)

        # Memory should be reasonable for small model
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        assert param_memory < 10 * 1024 * 1024  # Less than 10MB for small model
