"""
Training package for model training and inference.

This package provides trainer classes for pre-training and fine-tuning
GPT models, handling checkpointing, evaluation, and text generation.
"""

from .trainer import Trainer
from .fine_tuning_trainer import FineTuningTrainer
from .muon import Muon, create_muon_optimizer

__all__ = ["Trainer", "FineTuningTrainer", "Muon", "create_muon_optimizer"]
