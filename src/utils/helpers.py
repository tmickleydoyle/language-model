"""Utility functions and helpers for the language model package.

This module provides comprehensive utility functions including:
- Logging configuration and setup
- File validation and existence checking
- Model parameter counting and formatting
- Configuration serialization and deserialization
- Automatic configuration adjustment and compatibility fixes
- JSON handling with proper error management
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Tuple, TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration with console and file handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Raises:
        ValueError: If logging level is invalid
    """
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        raise ValueError(f"Invalid logging level: {level}")

    handlers = _create_logging_handlers()

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def _create_logging_handlers() -> list:
    """Create logging handlers for console and file output.

    Returns:
        List of logging handlers
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        handlers.append(logging.FileHandler('training.log'))
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not create file handler: {e}")

    return handlers


def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> bool:
    """Validate that a file exists and is accessible.

    Args:
        file_path: Path to the file to check
        description: Description for error messages

    Returns:
        True if file exists and is accessible

    Raises:
        FileNotFoundError: If file does not exist or is not accessible
        ValueError: If file_path is None or empty
    """
    _validate_file_path(file_path, description)
    _check_file_existence(file_path, description)
    _validate_file_type(file_path, description)
    return True


def _validate_file_path(file_path: Union[str, Path], description: str) -> None:
    """Validate that file path is not None or empty.

    Args:
        file_path: Path to validate
        description: Description for error messages

    Raises:
        ValueError: If file_path is None or empty
    """
    if file_path is None:
        raise ValueError(f"{description} path cannot be None")

    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    if not file_path_str.strip():
        raise ValueError(f"{description} path cannot be empty")


def _check_file_existence(file_path: Union[str, Path], description: str) -> None:
    """Check if file exists at the given path.

    Args:
        file_path: Path to check
        description: Description for error messages

    Raises:
        FileNotFoundError: If file does not exist
    """
    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    if not os.path.exists(file_path_str):
        raise FileNotFoundError(f"{description} not found: {file_path_str}")


def _validate_file_type(file_path: Union[str, Path], description: str) -> None:
    """Validate that path points to a file, not a directory.

    Args:
        file_path: Path to validate
        description: Description for error messages

    Raises:
        FileNotFoundError: If path is a directory
    """
    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    if os.path.isdir(file_path_str):
        raise FileNotFoundError(f"{description} is a directory, not a file: {file_path_str}")


def count_parameters(model: Any) -> Tuple[int, int]:
    """Count the total number of parameters and trainable parameters in a model.

    Args:
        model: PyTorch model with parameters() method

    Returns:
        Tuple of (total_parameters, trainable_parameters)

    Raises:
        AttributeError: If model doesn't have parameters() method
    """
    if not hasattr(model, 'parameters'):
        raise AttributeError("Model must have a parameters() method")

    total_params = _count_total_parameters(model)
    trainable_params = _count_trainable_parameters(model)

    return total_params, trainable_params


def _count_total_parameters(model: Any) -> int:
    """Count total parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def _count_trainable_parameters(model: Any) -> int:
    """Count trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(count: int) -> str:
    """Format parameter count for human-readable display.

    Args:
        count: Number of parameters

    Returns:
        Formatted string (e.g., "1.2M", "3.4K", "542")

    Raises:
        ValueError: If count is negative
    """
    if count < 0:
        raise ValueError("Parameter count cannot be negative")

    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    else:
        return str(count)


def save_config_to_dict(config: Any) -> Dict[str, Any]:
    """Convert a config object to a dictionary for serialization.

    Args:
        config: Configuration object with attributes

    Returns:
        Dictionary representation of config (non-private, non-callable attributes)

    Raises:
        ValueError: If config is None
    """
    if config is None:
        raise ValueError("Config cannot be None")

    return _extract_config_attributes(config)


def _extract_config_attributes(config: Any) -> Dict[str, Any]:
    """Extract non-private, non-callable attributes from config object.

    Args:
        config: Configuration object

    Returns:
        Dictionary of attribute names and values
    """
    return {
        attr: getattr(config, attr)
        for attr in dir(config)
        if not attr.startswith('_') and not callable(getattr(config, attr))
    }


def save_config_to_json(config: Any, file_path: Union[str, Path]) -> None:
    """Save a config object to a JSON file.

    Args:
        config: Configuration object to save
        file_path: Path to save the JSON file

    Raises:
        ValueError: If config or file_path is None/empty
        OSError: If unable to create directory or write file
        json.JSONEncodeError: If config cannot be serialized to JSON
    """
    _validate_save_config_params(config, file_path)
    config_dict = save_config_to_dict(config)
    _ensure_directory_exists(file_path)
    _write_config_json(config_dict, file_path)


def _validate_save_config_params(config: Any, file_path: Union[str, Path]) -> None:
    """Validate parameters for saving config to JSON.

    Args:
        config: Configuration object
        file_path: Path to save the file

    Raises:
        ValueError: If parameters are invalid
    """
    if config is None:
        raise ValueError("Config cannot be None")

    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    if not file_path_str or not file_path_str.strip():
        raise ValueError("File path cannot be None or empty")


def _ensure_directory_exists(file_path: Union[str, Path]) -> None:
    """Create directory for file path if it doesn't exist.

    Args:
        file_path: Path to the file

    Raises:
        OSError: If unable to create directory
    """
    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    directory = os.path.dirname(file_path_str)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _write_config_json(config_dict: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Write config dictionary to JSON file.

    Args:
        config_dict: Dictionary to write
        file_path: Path to write the file

    Raises:
        OSError: If unable to write file
        json.JSONEncodeError: If config cannot be serialized
    """
    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    with open(file_path_str, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_json(file_path: Union[str, Path]) -> 'Config':
    """Load configuration from a JSON file and return a Config object.

    Args:
        file_path: Path to the JSON file

    Returns:
        Config object containing configuration data

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON
        ValueError: If file_path is None/empty
    """
    validate_file_exists(file_path, "Configuration file")
    config_dict = _load_json_file(file_path)
    return _create_config_from_dict(config_dict)


def _load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary loaded from JSON

    Raises:
        json.JSONDecodeError: If file contains invalid JSON
        OSError: If unable to read file
    """
    # Convert pathlib.Path to string if necessary
    file_path_str = str(file_path)
    with open(file_path_str, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_config_from_dict(config_dict: Dict[str, Any]) -> 'Config':
    """Create Config object from dictionary.

    Args:
        config_dict: Dictionary containing config data

    Returns:
        Config object
    """
    # Import Config here to avoid circular imports
    from ..config import Config
    return Config.from_dict(config_dict)


def auto_adjust_config(config: Any) -> Any:
    """Auto-adjust configuration parameters to ensure compatibility.

    This function modifies a config object to fix common compatibility issues:
    - Adjusts n_head to ensure n_embd is divisible by n_head

    Args:
        config: Configuration object to adjust

    Returns:
        The modified config object

    Raises:
        ValueError: If config is None
    """
    if config is None:
        raise ValueError("Config cannot be None")

    _adjust_attention_heads(config)
    return config


def _adjust_attention_heads(config: Any) -> None:
    """Adjust attention heads to ensure embedding dimension divisibility.

    Args:
        config: Configuration object to modify
    """
    if not (hasattr(config, 'n_embd') and hasattr(config, 'n_head')):
        return

    if config.n_embd % config.n_head != 0:
        best_n_head = _find_best_n_head(config.n_embd, config.n_head)
        _log_n_head_adjustment(config.n_head, best_n_head, config.n_embd)
        config.n_head = best_n_head


def _find_best_n_head(n_embd: int, current_n_head: int) -> int:
    """Find the largest divisor of n_embd that's close to current n_head.

    Args:
        n_embd: Embedding dimension
        current_n_head: Current number of attention heads

    Returns:
        Best number of attention heads
    """
    for candidate in range(current_n_head, 0, -1):
        if n_embd % candidate == 0:
            return candidate
    return 1  # Fallback to 1 if no divisor found


def _log_n_head_adjustment(old_n_head: int, new_n_head: int, n_embd: int) -> None:
    """Log the n_head adjustment.

    Args:
        old_n_head: Original number of heads
        new_n_head: New number of heads
        n_embd: Embedding dimension
    """
    logger.warning(
        f"Auto-adjusted n_head from {old_n_head} to {new_n_head} "
        f"to ensure n_embd ({n_embd}) is divisible by n_head"
    )
