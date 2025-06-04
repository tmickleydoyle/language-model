"""
Model package for GPT (Generative Pre-trained Transformer) implementation.

This package provides the GPTLanguageModel class and related
transformer architecture components.
"""

from .gpt import (
    GPTLanguageModel,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    create_model
)

# Create alias for backwards compatibility with tests
Block = TransformerBlock


def create_model_factory(config, vocab_size):
    """
    Factory function to create GPT models.

    Args:
        config: Configuration object with model hyperparameters
        vocab_size (int): Size of the vocabulary

    Returns:
        GPTLanguageModel instance
    """
    # Update config with vocab_size
    config.vocab_size = vocab_size
    
    model = GPTLanguageModel(config)
    model = model.to(config.device)
    return model


__all__ = [
    "GPTLanguageModel",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "Block",  # Alias for backwards compatibility
    "create_model",
    "create_model_factory"
]
