"""Performance benchmark tests for the GPT language model.

These tests measure speed, memory usage, and throughput to ensure
the model architecture is fast and efficient, especially for tiny models.
"""
import time
import pytest
import torch

from src.config import Config
from src.model import create_model


class TestGenerationSpeed:
    """Test generation speed benchmarks."""

    def test_tiny_model_generation_speed(self):
        """Benchmark generation speed for tiny model (200K params)."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=256,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)
        model.eval()

        # Warmup
        idx = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            _ = model.generate(idx, max_new_tokens=10)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=50, temperature=1.0)
        elapsed = time.time() - start

        tokens_per_sec = 50 / elapsed
        print(f"\nTiny model: {tokens_per_sec:.1f} tokens/sec")

        # Should be reasonably fast on CPU (at least 100 tokens/sec)
        assert tokens_per_sec > 100, f"Too slow: {tokens_per_sec:.1f} tokens/sec"
        assert output.shape == (1, 60)

    def test_small_model_generation_speed(self):
        """Benchmark generation speed for small model (1.8M params)."""
        config = Config(
            n_embd=128,
            n_head=4,
            n_layer=4,
            block_size=512,
            vocab_size=5000,
            device='cpu'
        )
        model = create_model(config, 5000)
        model.eval()

        # Warmup
        idx = torch.randint(0, 5000, (1, 10))
        with torch.no_grad():
            _ = model.generate(idx, max_new_tokens=10)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=50, temperature=1.0)
        elapsed = time.time() - start

        tokens_per_sec = 50 / elapsed
        print(f"\nSmall model: {tokens_per_sec:.1f} tokens/sec")

        # Should be reasonably fast on CPU (at least 50 tokens/sec)
        assert tokens_per_sec > 50, f"Too slow: {tokens_per_sec:.1f} tokens/sec"
        assert output.shape == (1, 60)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_batch_generation_speed(self, batch_size):
        """Test generation speed with different batch sizes."""
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

        idx = torch.randint(0, 1000, (batch_size, 10))

        start = time.time()
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20)
        elapsed = time.time() - start

        total_tokens = batch_size * 20
        tokens_per_sec = total_tokens / elapsed
        print(f"\nBatch {batch_size}: {tokens_per_sec:.1f} tokens/sec")

        assert output.shape == (batch_size, 30)


class TestTrainingSpeed:
    """Test training speed benchmarks."""

    def test_tiny_model_training_step_speed(self):
        """Benchmark single training step for tiny model."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu',
            batch_size=16
        )
        model = create_model(config, 1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        x = torch.randint(0, 1000, (16, 32))
        y = torch.randint(0, 1000, (16, 32))
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Benchmark
        optimizer.zero_grad()
        start = time.time()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - start

        tokens_per_sec = (16 * 32) / elapsed
        print(f"\nTraining: {tokens_per_sec:.1f} tokens/sec, {elapsed*1000:.1f}ms/step")

        # Should process at least 1000 tokens/sec
        assert tokens_per_sec > 1000, f"Too slow: {tokens_per_sec:.1f} tokens/sec"


class TestMemoryUsage:
    """Test memory usage benchmarks."""

    def test_tiny_model_memory_footprint(self):
        """Test memory footprint of tiny model."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=256,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Calculate parameter memory
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = (total_params * 4) / (1024 ** 2)  # FP32

        print(f"\nTiny model: {total_params:,} params, {param_memory_mb:.2f}MB")

        # Tiny model should be under 1MB
        assert param_memory_mb < 1.0, f"Too large: {param_memory_mb:.2f}MB"
        assert total_params < 250000, f"Too many params: {total_params:,}"

    def test_small_model_memory_footprint(self):
        """Test memory footprint of small model."""
        config = Config(
            n_embd=128,
            n_head=4,
            n_layer=4,
            block_size=512,
            vocab_size=5000,
            device='cpu'
        )
        model = create_model(config, 5000)

        # Calculate parameter memory
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = (total_params * 4) / (1024 ** 2)  # FP32

        print(f"\nSmall model: {total_params:,} params, {param_memory_mb:.2f}MB")

        # Small model should be under 10MB
        assert param_memory_mb < 10.0, f"Too large: {param_memory_mb:.2f}MB"
        assert total_params < 2500000, f"Too many params: {total_params:,}"

    def test_gqa_memory_savings(self):
        """Test that GQA reduces memory compared to standard attention."""
        config = Config(
            n_embd=128,
            n_head=8,
            n_layer=2,
            block_size=256,
            vocab_size=1000,
            device='cpu',
            n_kv_head=2  # 1/4 of query heads
        )
        model = create_model(config, 1000)

        # Count attention parameters
        attn_params = 0
        for module in model.modules():
            if hasattr(module, 'k_proj'):
                attn_params += sum(p.numel() for p in [
                    module.k_proj.weight,
                    module.v_proj.weight
                ])

        print(f"\nGQA attention params: {attn_params:,}")

        # GQA should use significantly fewer KV parameters
        # Standard would be 2 * (n_embd * n_embd) per layer
        standard_kv_params = 2 * (128 * 128) * 2
        assert attn_params < standard_kv_params * 0.5, "GQA not saving memory"


class TestParameterEfficiency:
    """Test parameter efficiency for tiny models."""

    def test_tiny_model_parameter_distribution(self):
        """Test parameter distribution in tiny model."""
        config = Config(
            n_embd=64,
            n_head=4,
            n_layer=2,
            block_size=128,
            vocab_size=1000,
            device='cpu'
        )
        model = create_model(config, 1000)

        # Count parameters by component
        embedding_params = model.token_embedding.weight.numel()
        lm_head_params = 0  # Tied weights

        total_params = sum(p.numel() for p in model.parameters())
        non_embedding_params = total_params - embedding_params

        print(f"\nTotal: {total_params:,}")
        print(f"Embeddings: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
        print(f"Transformer: {non_embedding_params:,} ({non_embedding_params/total_params*100:.1f}%)")

        # For tiny models, embeddings shouldn't dominate
        assert embedding_params / total_params < 0.5, "Embeddings too large"

    def test_model_scales_linearly(self):
        """Test that model parameters scale linearly with size."""
        configs = [
            (64, 2, "tiny"),
            (128, 4, "small"),
            (256, 6, "medium")
        ]

        param_counts = []
        for n_embd, n_layer, name in configs:
            config = Config(
                n_embd=n_embd,
                n_head=4,
                n_layer=n_layer,
                block_size=256,
                vocab_size=1000,
                device='cpu'
            )
            model = create_model(config, 1000)
            params = sum(p.numel() for p in model.parameters())
            param_counts.append(params)
            print(f"\n{name}: {params:,} params")

        # Larger models should have more parameters (roughly linear scaling)
        assert param_counts[0] < param_counts[1] < param_counts[2]

        # Check approximate linear scaling (within 2x factor)
        ratio_1_0 = param_counts[1] / param_counts[0]
        ratio_2_1 = param_counts[2] / param_counts[1]
        assert 2 < ratio_1_0 < 10
        assert 2 < ratio_2_1 < 10


class TestGenerationQuality:
    """Test generation quality metrics."""

    def test_generation_produces_valid_tokens(self):
        """Test that generation produces valid tokens."""
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

        # All generated tokens should be valid
        assert torch.all(output >= 0)
        assert torch.all(output < 1000)

        # Should generate different tokens (not all the same)
        unique_tokens = torch.unique(output).numel()
        assert unique_tokens > 5, "Generated too few unique tokens"

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_temperature_affects_diversity(self, temperature):
        """Test that temperature affects generation diversity."""
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
            output = model.generate(idx, max_new_tokens=50, temperature=temperature)

        unique_tokens = torch.unique(output).numel()
        print(f"\nTemp {temperature}: {unique_tokens} unique tokens")

        # Higher temperature should generally produce more diversity
        assert unique_tokens > 1
        assert output.shape == (1, 60)
