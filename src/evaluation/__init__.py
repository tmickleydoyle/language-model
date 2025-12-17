"""Evaluation utilities for language model quality assessment.

This module provides tools for measuring language model quality:
- Perplexity computation
- Diversity metrics
- Syntax validation (for code models)
- Completion benchmarks
"""

from .metrics import (
    compute_perplexity,
    compute_diversity_score,
    compute_token_accuracy,
    compute_repetition_score,
)
from .benchmarks import (
    run_syntax_validation,
    run_completion_benchmark,
    run_repetition_analysis,
    evaluate_model_quality,
)

__all__ = [
    "compute_perplexity",
    "compute_diversity_score",
    "compute_token_accuracy",
    "compute_repetition_score",
    "run_syntax_validation",
    "run_completion_benchmark",
    "run_repetition_analysis",
    "evaluate_model_quality",
]
