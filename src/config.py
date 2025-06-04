"""
Configuration module for the GPT language model.

This module provides the Config class that centralizes all hyperparameters,
system settings, and logging configuration for the model training and inference.
"""

import logging
import os
import torch
from dataclasses import dataclass
from typing import Optional, Any, Dict


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs

    Returns:
        Configured logger instance

    Raises:
        ValueError: If level is invalid
    """
    # Get the root logger
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    # Set level - validate it exists
    if not hasattr(logging, level.upper()):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Centralized configuration class for all model variants and training settings.

    This class uses dataclasses for better maintainability and type safety.
    All hyperparameters are documented with their purpose and typical ranges.

    Model Architecture:
        n_embd: Embedding dimension (typical: 256-1024)
        n_head: Number of attention heads (typical: 4-16)
        n_layer: Number of transformer blocks (typical: 4-12)
        block_size: Maximum context length (typical: 128-2048)
        dropout: Dropout rate for regularization (typical: 0.0-0.2)

    Training:
        batch_size: Number of sequences processed in parallel
        max_iters: Maximum training iterations
        max_epochs: Maximum training epochs
        learning_rate: Adam optimizer learning rate
        weight_decay: L2 regularization weight decay
        eval_interval: Steps between evaluations
        eval_iters: Number of evaluation iterations
        save_interval: Steps between model saves
        max_batches_per_epoch: Maximum batches to process per epoch

    Generation:
        default_max_tokens: Default maximum tokens for text generation
        default_temperature: Default temperature for text generation
        default_top_k: Default top-k sampling value

    Data:
        vocab_size: BPE vocabulary size

    System:
        device: Computation device ('cuda', 'mps', or 'cpu')
        seed: Random seed for reproducibility
        fp16: Whether to use mixed precision training

    Logging:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
    """

    # Model architecture hyperparameters
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 64
    max_iters: int = 10000
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01  # Centralized weight decay value
    eval_interval: int = 100
    eval_iters: int = 100
    save_interval: int = 100
    max_batches_per_epoch: int = 100  # Centralized batch limit per epoch
    
    # Fine-tuning specific parameters
    grad_clip: float = 1.0  # Gradient clipping threshold
    scheduler_type: str = "cosine"  # Learning rate scheduler type

    # Generation parameters
    default_max_tokens: int = 100  # Centralized default max tokens
    default_temperature: float = 1.0  # Centralized default temperature
    default_top_k: Optional[int] = None  # Centralized default top-k

    # Data parameters
    vocab_size: int = 50257

    # System configuration
    device: Optional[str] = None
    seed: int = 1337
    fp16: bool = False

    # Logging configuration
    log_level: str = "INFO"  # Centralized logging level
    log_file: Optional[str] = None  # Centralized log file path

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Config with backward compatibility."""
        # Handle backward compatibility for parameter names - don't modify kwargs
        processed_kwargs = kwargs.copy()

        # Apply dataclass __init__ with processed kwargs
        self.__dataclass_init__(**processed_kwargs)

    def __dataclass_init__(self, **kwargs: Any) -> None:
        """Apply dataclass initialization logic."""
        for field_name, field_obj in self.__dataclass_fields__.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                # Use default value
                setattr(self, field_name, field_obj.default)

        # Call post_init manually since we're overriding __init__
        self.__post_init__()

    def __post_init__(self) -> None:
        """Initialize derived and device-dependent settings."""
        # Set device if not specified or if auto
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Setup logging with centralized configuration
        setup_logging(level=self.log_level, log_file=self.log_file)

        # Validate configuration
        self._validate_config()

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        logger.info(f"Config initialized with device: {self.device}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")

        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")

        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {self.max_iters}")

        if self.eval_interval <= 0:
            raise ValueError(
                f"eval_interval must be positive, got {self.eval_interval}"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if self.max_batches_per_epoch <= 0:
            raise ValueError(
                f"max_batches_per_epoch must be positive, got {self.max_batches_per_epoch}"
            )

        if self.default_max_tokens <= 0:
            raise ValueError(
                f"default_max_tokens must be positive, got {self.default_max_tokens}"
            )

        if self.default_temperature <= 0:
            raise ValueError(
                f"default_temperature must be positive, got {self.default_temperature}"
            )

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head"
            )

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")

        # Validate log level
        if not hasattr(logging, self.log_level.upper()):
            raise ValueError(f"Invalid log level: {self.log_level}")

    @property
    def head_dim(self) -> int:
        """Calculate dimension per attention head."""
        return self.n_embd // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        from dataclasses import fields
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = value
        # Add computed properties that tests expect
        result['head_dim'] = self.head_dim
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        # Remove computed properties that shouldn't be passed to constructor
        config_dict = config_dict.copy()
        config_dict.pop('head_dim', None)

        # Handle special case where n_embd is provided but n_head isn't
        # Ensure compatibility by adjusting n_head if needed
        if 'n_embd' in config_dict and 'n_head' not in config_dict:
            n_embd = config_dict['n_embd']
            default_n_head = 6  # Default value from dataclass

            # First check if default value works
            if n_embd % default_n_head == 0:
                config_dict['n_head'] = default_n_head
            # Otherwise choose a compatible n_head value
            elif n_embd == 512:
                config_dict['n_head'] = 8  # 512 is divisible by 8
            elif n_embd == 768:
                config_dict['n_head'] = 12  # 768 is divisible by 12
            elif n_embd == 1024:
                config_dict['n_head'] = 16  # 1024 is divisible by 16
            # For other values, try to find a good divisor
            else:
                for head_count in [8, 4, 12, 16]:
                    if n_embd % head_count == 0:
                        config_dict['n_head'] = head_count
                        break

        return cls(**config_dict)

    def save(self, file_path: str) -> None:
        """Save config to JSON file."""
        from .utils.helpers import save_config_to_json
        save_config_to_json(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'Config':
        """Load config from JSON file."""
        from .utils.helpers import load_config_from_json
        config_instance = load_config_from_json(file_path)
        return config_instance

    def get_device(self) -> str:
        """Get the configured device."""
        if self.device is None:
            return 'cpu'
        return self.device
