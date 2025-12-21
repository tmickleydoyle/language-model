"""
Model package for GPT (Generative Pre-trained Transformer) implementation.

This package provides both classic and modern GPT implementations with
related transformer architecture components.
"""

from .gpt_classic import (
    GPTLanguageModel as ClassicGPTLanguageModel,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    create_model as create_classic_model
)

from .gpt import (
    ModernGPTLanguageModel,
    GPTLanguageModel,  # Main export - now points to modern implementation
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    SlidingWindowAttention,
    ContextualPositionalEmbedding,
    MixtureOfExperts,
    xSwiGLU,
    SwiGLU,
    RMSNorm,
    RotaryPositionalEmbedding,
    ModernTransformerBlock,
    create_model,
    create_modern_model,
    create_optimal_model
)

from .recursive import (
    RecursiveGPTLanguageModel,
    RecursiveTransformerBlock,
    ConfidenceHead,
    create_recursive_model
)

# Use modern implementation as default
GPTLanguageModel = ModernGPTLanguageModel
create_model = create_modern_model

# Backwards compatibility alias for tests (DO NOT USE IN NEW CODE)
# Modern transformers should use ModernTransformerBlock directly
Block = ModernTransformerBlock


def create_model_factory(config, vocab_size):
    """
    Factory function to create GPT models.

    Args:
        config: Configuration object with model hyperparameters
        vocab_size (int): Size of the vocabulary

    Returns:
        ModernGPTLanguageModel instance
    """
    return create_model(config, vocab_size)


__all__ = [
    # Main model classes
    "GPTLanguageModel",  # Modern by default
    "ModernGPTLanguageModel",
    "ClassicGPTLanguageModel",
    "RecursiveGPTLanguageModel",  # Recursive reasoning variant

    # Modern attention mechanisms
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "SlidingWindowAttention",

    # Position encodings
    "RotaryPositionalEmbedding",
    "ContextualPositionalEmbedding",

    # Feedforward networks
    "xSwiGLU",  # Modern default
    "SwiGLU",
    "MixtureOfExperts",

    # Recursive reasoning components
    "RecursiveTransformerBlock",
    "ConfidenceHead",

    # Other components
    "RMSNorm",
    "ModernTransformerBlock",

    # Classic components (for reference)
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "Block",  # Alias for backwards compatibility

    # Factory functions
    "create_model",  # Modern by default
    "create_modern_model",
    "create_optimal_model",  # Best defaults
    "create_classic_model",
    "create_recursive_model",  # Recursive variant
    "create_model_factory"
]
