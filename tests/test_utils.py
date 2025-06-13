"""Tests for the utils module."""
import pytest
import logging
import json

from src.utils import (
    validate_file_exists,
    count_parameters,
    save_config_to_json,
    load_config_from_json
)
from src.config import Config, setup_logging


class TestUtils:
    """Test cases for utility functions."""

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        setup_logging()

        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

        # Check that we can log messages
        logger = logging.getLogger(__name__)
        logger.info("Test message")  # Should not raise any errors

    def test_setup_logging_debug_level(self):
        """Test setup_logging with DEBUG level."""
        setup_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, temp_dir):
        """Test setup_logging with log file."""
        log_file = temp_dir / "test.log"
        setup_logging(level="INFO", log_file=log_file)

        # Log a message
        logger = logging.getLogger(__name__)
        test_message = "Test log message"
        logger.info(test_message)

        # Check that file was created and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID")

    def test_setup_logging_file_creation(self, temp_dir):
        """Test that log file directory is created if it doesn't exist."""
        nested_dir = temp_dir / "nested" / "log" / "dir"
        log_file = nested_dir / "test.log"

        setup_logging(log_file=log_file)

        # Log a message to trigger file creation
        logger = logging.getLogger(__name__)
        logger.info("Test message")

        assert nested_dir.exists()
        assert log_file.exists()

    def test_validate_file_exists_valid_file(self, temp_dir):
        """Test validate_file_exists with existing file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Should not raise any exception
        validate_file_exists(test_file, "Test file")

    def test_validate_file_exists_nonexistent_file(self, temp_dir):
        """Test validate_file_exists with non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            validate_file_exists(nonexistent_file, "Test file")

    def test_validate_file_exists_directory(self, temp_dir):
        """Test validate_file_exists with directory instead of file."""
        with pytest.raises(FileNotFoundError, match="Test dir is a directory, not a file"):
            validate_file_exists(temp_dir, "Test dir")

    def test_validate_file_exists_none_path(self):
        """Test validate_file_exists with None path."""
        with pytest.raises(ValueError, match="Test file path cannot be None"):
            validate_file_exists(None, "Test file")

    def test_count_parameters_simple_model(self):
        """Test count_parameters with simple model."""
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 5),  # 10*5 + 5 = 55 parameters
            nn.Linear(5, 1)    # 5*1 + 1 = 6 parameters
        )
        # Total: 61 parameters

        total_params, trainable_params = count_parameters(model)
        assert total_params == 61
        assert trainable_params == 61

    def test_count_parameters_frozen_model(self):
        """Test count_parameters with some frozen parameters."""
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 5),  # 55 parameters
            nn.Linear(5, 1)    # 6 parameters
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        total_params, trainable_params = count_parameters(model)
        assert total_params == 61
        assert trainable_params == 6  # Only second layer is trainable

    def test_count_parameters_empty_model(self):
        """Test count_parameters with model with no parameters."""
        import torch.nn as nn

        model = nn.Sequential()  # Empty model

        total_params, trainable_params = count_parameters(model)
        assert total_params == 0
        assert trainable_params == 0

    def test_count_parameters_gpt_model(self, small_model_config):
        """Test count_parameters with GPT model."""
        from src.model import GPTLanguageModel

        model = GPTLanguageModel(small_model_config)
        total_params, trainable_params = count_parameters(model)

        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

        # Rough estimate check (should be reasonable for small model)
        assert total_params < 1_000_000  # Less than 1M parameters for small model

    def test_save_config_to_json(self, temp_dir, sample_config):
        """Test save_config_to_json function."""
        config_file = temp_dir / "config.json"

        save_config_to_json(sample_config, config_file)

        assert config_file.exists()

        # Verify content
        with open(config_file, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["vocab_size"] == sample_config.vocab_size
        assert saved_data["n_embd"] == sample_config.n_embd
        assert saved_data["device"] == sample_config.device

    def test_save_config_to_json_create_directory(self, temp_dir, sample_config):
        """Test that save_config_to_json creates directory if needed."""
        nested_dir = temp_dir / "nested" / "config" / "dir"
        config_file = nested_dir / "config.json"

        save_config_to_json(sample_config, config_file)

        assert nested_dir.exists()
        assert config_file.exists()

    def test_save_config_to_json_overwrite(self, temp_dir, sample_config):
        """Test that save_config_to_json overwrites existing file."""
        config_file = temp_dir / "config.json"

        # Create existing file with different content
        config_file.write_text('{"old": "data"}')

        save_config_to_json(sample_config, config_file)

        # Should have overwritten
        with open(config_file, 'r') as f:
            saved_data = json.load(f)

        assert "old" not in saved_data
        assert saved_data["vocab_size"] == sample_config.vocab_size

    def test_load_config_from_json(self, temp_dir, sample_config):
        """Test load_config_from_json function."""
        config_file = temp_dir / "config.json"

        # Save config first
        save_config_to_json(sample_config, config_file)

        # Load it back
        loaded_config = load_config_from_json(config_file)

        assert isinstance(loaded_config, Config)
        assert loaded_config.vocab_size == sample_config.vocab_size
        assert loaded_config.n_embd == sample_config.n_embd
        assert loaded_config.device == sample_config.device

    def test_load_config_from_json_nonexistent(self, temp_dir):
        """Test load_config_from_json with non-existent file."""
        config_file = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_config_from_json(config_file)

    def test_load_config_from_json_invalid_json(self, temp_dir):
        """Test load_config_from_json with invalid JSON."""
        config_file = temp_dir / "invalid.json"
        config_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_config_from_json(config_file)

    def test_load_config_from_json_partial_config(self, temp_dir):
        """Test load_config_from_json with partial configuration."""
        config_file = temp_dir / "partial_config.json"

        # Save only some fields
        partial_data = {
            "vocab_size": 2000,
            "n_embd": 512,
            "learning_rate": 1e-3
        }

        with open(config_file, 'w') as f:
            json.dump(partial_data, f)

        loaded_config = load_config_from_json(config_file)

        # Should have specified values
        assert loaded_config.vocab_size == 2000
        assert loaded_config.n_embd == 512
        assert loaded_config.learning_rate == 1e-3

        # Should have defaults for unspecified values
        # n_head is auto-adjusted to 8 for n_embd=512 to ensure divisibility
        assert loaded_config.n_head == 8  # Auto-adjusted for n_embd=512
        assert loaded_config.n_layer == 6  # Default value

    def test_config_roundtrip_json(self, temp_dir):
        """Test complete roundtrip: config -> JSON -> config."""
        original_config = Config(
            vocab_size=5000,
            n_embd=256,
            n_head=8,
            n_layer=6,
            block_size=512,
            dropout=0.2,
            learning_rate=5e-4,
            batch_size=16,
            max_iters=500,
            eval_interval=25,
            eval_iters=10,
            device="cuda"
        )

        config_file = temp_dir / "roundtrip_config.json"

        # Save and load
        save_config_to_json(original_config, config_file)
        loaded_config = load_config_from_json(config_file)

        # Should be identical
        assert loaded_config == original_config

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_setup_logging_all_levels(self, log_level, temp_dir):
        """Test setup_logging with all valid log levels."""
        log_file = temp_dir / f"test_{log_level.lower()}.log"
        setup_logging(level=log_level, log_file=log_file)

        expected_level = getattr(logging, log_level)

        # Root logger should be set to the specified level
        root_logger = logging.getLogger()
        assert root_logger.level == expected_level

    def test_logging_formatting(self, temp_dir):
        """Test that log messages are properly formatted."""
        log_file = temp_dir / "format_test.log"
        setup_logging(level="INFO", log_file=log_file)

        logger = logging.getLogger("test_module")
        test_message = "Test formatting message"
        logger.info(test_message)

        log_content = log_file.read_text()

        # Should contain timestamp, level, module name, and message
        assert test_message in log_content
        assert "INFO" in log_content
        assert "test_module" in log_content
        # Should have timestamp (basic check for date format)
        import re
        assert re.search(r'\d{4}-\d{2}-\d{2}', log_content)

    def test_validate_file_exists_with_pathlib(self, temp_dir):
        """Test validate_file_exists with pathlib.Path objects."""
        test_file = temp_dir / "pathlib_test.txt"
        test_file.write_text("test")

        # Should work with Path objects
        validate_file_exists(test_file, "Pathlib file")

    def test_validate_file_exists_with_string(self, temp_dir):
        """Test validate_file_exists with string paths."""
        test_file = temp_dir / "string_test.txt"
        test_file.write_text("test")

        # Should work with string paths
        validate_file_exists(str(test_file), "String file")

    def test_utils_error_propagation(self, temp_dir):
        """Test that utilities properly propagate errors."""
        # Test file operations with permission errors (if possible)
        import os
        import stat

        if os.name != 'nt':  # Skip on Windows due to permission model differences
            restricted_dir = temp_dir / "restricted"
            restricted_dir.mkdir()

            # Remove write permissions
            restricted_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

            try:
                restricted_file = restricted_dir / "test.txt"

                # This should raise a permission error
                with pytest.raises(Exception):  # Could be PermissionError or OSError
                    restricted_file.write_text("test")
            finally:
                # Restore permissions for cleanup
                restricted_dir.chmod(stat.S_IRWXU)

    def test_parameter_counting_edge_cases(self):
        """Test count_parameters with edge cases."""
        import torch.nn as nn

        # Model with shared parameters
        embedding = nn.Embedding(100, 50)
        model = nn.ModuleDict({
            'embed1': embedding,
            'embed2': embedding,  # Same embedding used twice
            'linear': nn.Linear(50, 10)
        })

        total_params, trainable_params = count_parameters(model)

        # Should count shared parameters only once
        expected_embed_params = 100 * 50  # embedding parameters
        expected_linear_params = 50 * 10 + 10  # linear parameters
        expected_total = expected_embed_params + expected_linear_params

        assert total_params == expected_total
        assert trainable_params == expected_total
