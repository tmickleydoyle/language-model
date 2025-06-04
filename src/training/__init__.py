"""
Training package for model training and inference.

This package provides trainer classes for pre-training and fine-tuning
GPT models, handling checkpointing, evaluation, and text generation.
"""

from .trainer import Trainer
from .fine_tuning_trainer import FineTuningTrainer

__all__ = ["Trainer", "FineTuningTrainer"]
