"""Tests for the config module."""
import pytest
import json
from dataclasses import asdict

from src.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = Config()
        assert config.vocab_size == 50257
        assert config.n_embd == 384
        assert config.n_head == 6
        assert config.n_layer == 6
        assert config.block_size == 256
        assert config.dropout == 0.1
        assert config.learning_rate == 3e-4
        assert config.batch_size == 64
        assert config.max_iters == 100000
        assert config.eval_interval == 100
        assert config.eval_iters == 100
        # Device is auto-detected, so check it's one of the valid options
        assert config.device in ["cuda", "mps", "cpu"]

    def test_custom_config_creation(self):
        """Test creating a config with custom values."""
        config = Config(
            vocab_size=1000,
            n_embd=256,
            n_head=8,
            n_layer=6,
            block_size=512,
            dropout=0.2,
            learning_rate=1e-4,
            batch_size=32,
            max_iters=50,
            eval_interval=50,
            eval_iters=25,
            device="cpu"
        )
        assert config.vocab_size == 1000
        assert config.n_embd == 256
        assert config.n_head == 8
        assert config.n_layer == 6
        assert config.block_size == 512
        assert config.dropout == 0.2
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.max_iters == 50
        assert config.eval_interval == 50
        assert config.eval_iters == 25
        assert config.device == "cpu"

    def test_config_validation_valid_values(self):
        """Test that valid values pass validation."""
        config = Config(
            vocab_size=1000,
            n_embd=256,
            n_head=8,
            n_layer=6,
            block_size=512,
            dropout=0.0,
            learning_rate=1e-3,
            batch_size=1,
            max_epochs=1,
            eval_interval=1,
            save_interval=1
        )
        # Should not raise any exceptions
        assert config.vocab_size == 1000

    def test_config_validation_invalid_vocab_size(self):
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Config(vocab_size=0)

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Config(vocab_size=-1)

    def test_config_validation_invalid_n_embd(self):
        """Test that invalid n_embd raises ValueError."""
        with pytest.raises(ValueError, match="n_embd must be positive"):
            Config(n_embd=0)

        with pytest.raises(ValueError, match="n_embd must be positive"):
            Config(n_embd=-1)

    def test_config_validation_invalid_n_head(self):
        """Test that invalid n_head raises ValueError."""
        with pytest.raises(ValueError, match="n_head must be positive"):
            Config(n_head=0)

        with pytest.raises(ValueError, match="n_head must be positive"):
            Config(n_head=-1)

    def test_config_validation_invalid_n_layer(self):
        """Test that invalid n_layer raises ValueError."""
        with pytest.raises(ValueError, match="n_layer must be positive"):
            Config(n_layer=0)

        with pytest.raises(ValueError, match="n_layer must be positive"):
            Config(n_layer=-1)

    def test_config_validation_invalid_block_size(self):
        """Test that invalid block_size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be positive"):
            Config(block_size=0)

        with pytest.raises(ValueError, match="block_size must be positive"):
            Config(block_size=-1)

    def test_config_validation_invalid_dropout(self):
        """Test that invalid dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            Config(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            Config(dropout=1.1)

    def test_config_validation_invalid_learning_rate(self):
        """Test that invalid learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            Config(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            Config(learning_rate=-1e-4)

    def test_config_validation_invalid_batch_size(self):
        """Test that invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            Config(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            Config(batch_size=-1)

    def test_config_validation_invalid_max_iters(self):
        """Test that invalid max_iters raises ValueError."""
        with pytest.raises(ValueError, match="max_iters must be positive"):
            Config(max_iters=0)

        with pytest.raises(ValueError, match="max_iters must be positive"):
            Config(max_iters=-1)

    def test_config_validation_invalid_eval_interval(self):
        """Test that invalid eval_interval raises ValueError."""
        with pytest.raises(ValueError, match="eval_interval must be positive"):
            Config(eval_interval=0)

        with pytest.raises(ValueError, match="eval_interval must be positive"):
            Config(eval_interval=-1)

    def test_config_validation_invalid_eval_iters(self):
        """Test that invalid eval_iters raises ValueError."""
        with pytest.raises(ValueError, match="eval_interval must be positive"):
            Config(eval_interval=0)

        with pytest.raises(ValueError, match="eval_interval must be positive"):
            Config(eval_interval=-1)

    def test_config_validation_n_embd_divisible_by_n_head(self):
        """Test that n_embd must be divisible by n_head."""
        with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
            Config(n_embd=100, n_head=7)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(vocab_size=1000, n_embd=240, n_head=6)  # 240 is divisible by 6
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["vocab_size"] == 1000
        assert config_dict["n_embd"] == 240
        assert "device" in config_dict

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vocab_size": 1000,
            "n_embd": 256,
            "n_head": 8,
            "n_layer": 6,
            "block_size": 512,
            "dropout": 0.2,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_epochs": 50,
            "eval_interval": 50,
            "save_interval": 500,
            "device": "cpu"
        }

        config = Config.from_dict(config_dict)
        assert config.vocab_size == 1000
        assert config.n_embd == 256
        assert config.n_head == 8
        assert config.device == "cpu"

    def test_config_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        config_dict = {
            "vocab_size": 1000,
            "n_embd": 240,  # Must be divisible by default n_head=6
        }

        config = Config.from_dict(config_dict)
        assert config.vocab_size == 1000
        assert config.n_embd == 240
        # Should use defaults for other values
        assert config.n_head == 6
        assert config.n_layer == 6

    def test_config_save_and_load(self, temp_dir):
        """Test saving and loading config to/from file."""
        config = Config(
            vocab_size=1000,
            n_embd=256,
            n_head=8,
            device="cpu"
        )

        config_file = temp_dir / "config.json"
        config.save(config_file)

        assert config_file.exists()

        # Load and verify
        loaded_config = Config.load(config_file)
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.n_embd == config.n_embd
        assert loaded_config.n_head == config.n_head
        assert loaded_config.device == config.device

    def test_config_save_load_roundtrip(self, temp_dir):
        """Test that save/load roundtrip preserves all values."""
        original_config = Config(
            vocab_size=2048,
            n_embd=512,
            n_head=16,
            n_layer=8,
            block_size=256,
            dropout=0.15,
            learning_rate=5e-4,
            batch_size=16,
            max_epochs=25,
            eval_interval=75,
            save_interval=250,
            device="cuda"
        )

        config_file = temp_dir / "roundtrip_config.json"
        original_config.save(config_file)
        loaded_config = Config.load(config_file)

        # Compare all fields
        assert asdict(original_config) == asdict(loaded_config)

    def test_config_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent_config.json")

    def test_config_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises appropriate error."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            Config.load(invalid_file)

    def test_config_repr(self):
        """Test string representation of config."""
        config = Config(vocab_size=1000)
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "vocab_size=1000" in repr_str

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = Config(vocab_size=1000, n_embd=240, n_head=6)  # 240 is divisible by 6
        config2 = Config(vocab_size=1000, n_embd=240, n_head=6)
        config3 = Config(vocab_size=2000, n_embd=240, n_head=6)

        assert config1 == config2
        assert config1 != config3

    def test_config_device_auto_resolution(self):
        """Test that 'auto' device gets resolved appropriately."""
        import torch

        config = Config(device="auto")
        resolved_device = config.get_device()

        # Should resolve to either 'cuda', 'mps', or 'cpu'
        assert resolved_device in ["cuda", "mps", "cpu"]

        if torch.cuda.is_available():
            assert resolved_device == "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            assert resolved_device == "mps"
        else:
            assert resolved_device == "cpu"

    def test_config_get_device_explicit(self):
        """Test get_device with explicit device setting."""
        config = Config(device="cpu")
        assert config.get_device() == "cpu"

        config = Config(device="cuda")
        assert config.get_device() == "cuda"
