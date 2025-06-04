"""Utility functions and helpers for the language model package."""

import json
import logging
import os
import sys
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )


def validate_file_exists(file_path: str, description: str = "File") -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file to check
        description: Description for error messages

    Returns:
        True if file exists

    Raises:
        FileNotFoundError: If file does not exist
    """
    if file_path is None:
        raise FileNotFoundError(f"{description} not found")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")
    if os.path.isdir(file_path):
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return True


def count_parameters(model: Any) -> Tuple[int, int]:
    """
    Count the total number of parameters and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_parameter_count(count: int) -> str:
    """
    Format parameter count for human-readable display.

    Args:
        count: Number of parameters

    Returns:
        Formatted string (e.g., "1.2M", "3.4K")
    """
    if count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    else:
        return str(count)


def save_config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert a config object to a dictionary for serialization.

    Args:
        config: Configuration object

    Returns:
        Dictionary representation of config
    """
    return {
        attr: getattr(config, attr)
        for attr in dir(config)
        if not attr.startswith('_') and not callable(getattr(config, attr))
    }


def save_config_to_json(config: Any, file_path: str) -> None:
    """
    Save a config object to a JSON file.

    Args:
        config: Configuration object
        file_path: Path to save the JSON file
    """
    config_dict = save_config_to_dict(config)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)


def load_config_from_json(file_path: str) -> 'Config':
    """
    Load configuration from a JSON file and return a Config object.

    Args:
        file_path: Path to the JSON file

    Returns:
        Config object containing configuration data

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    validate_file_exists(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Import Config here to avoid circular imports
    from ..config import Config
    return Config.from_dict(config_dict)


def auto_adjust_config(config: Any) -> Any:
    """
    Auto-adjust configuration parameters to ensure compatibility.

    This function modifies a config object to fix common compatibility issues:
    - Adjusts n_head to ensure n_embd is divisible by n_head

    Args:
        config: Configuration object to adjust

    Returns:
        The modified config object
    """
    logger = __import__('logging').getLogger(__name__)

    # Auto-adjust n_head if needed to ensure divisibility
    if hasattr(config, 'n_embd') and hasattr(config, 'n_head'):
        if config.n_embd % config.n_head != 0:
            # Find the largest divisor of n_embd that's close to n_head
            best_n_head = config.n_head
            for candidate in range(config.n_head, 0, -1):
                if config.n_embd % candidate == 0:
                    best_n_head = candidate
                    break

            if best_n_head != config.n_head:
                logger.warning(
                    f"Auto-adjusted n_head from {
                        config.n_head} to {best_n_head} to ensure n_embd ({
                        config.n_embd}) is divisible by n_head")
                config.n_head = best_n_head

    return config
