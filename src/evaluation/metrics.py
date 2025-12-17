"""Core evaluation metrics for language models.

This module provides metrics for evaluating language model quality:
- Perplexity: Standard measure of model uncertainty
- Diversity: Measures variety in generated text
- Accuracy: Token-level prediction accuracy
- Repetition: Detects degenerate repetitive patterns
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch


def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity.

    Perplexity = exp(loss) measures how "surprised" the model is.
    Lower is better:
    - < 20: Excellent for domain-specific models
    - 20-50: Good for small models
    - 50-100: Acceptable
    - > 100: Model is struggling

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value (capped at 10000 to prevent overflow)
    """
    capped_loss = min(loss, 10.0)
    return math.exp(capped_loss)


def compute_diversity_score(samples: List[str]) -> Dict[str, float]:
    """Measure diversity of generated samples.

    Diversity metrics indicate how varied and non-repetitive the
    model's outputs are. Higher values are better.

    Args:
        samples: List of generated text samples

    Returns:
        Dictionary with:
        - unique_1gram: Ratio of unique unigrams
        - unique_2gram: Ratio of unique bigrams
        - unique_3gram: Ratio of unique trigrams
        - diversity_score: Combined diversity metric (0-1)
    """
    if not samples:
        return {
            "unique_1gram": 0.0,
            "unique_2gram": 0.0,
            "unique_3gram": 0.0,
            "diversity_score": 0.0,
        }

    all_words: List[str] = []
    all_bigrams: List[Tuple[str, str]] = []
    all_trigrams: List[Tuple[str, str, str]] = []

    for sample in samples:
        words = sample.split()
        all_words.extend(words)

        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i + 1]))

        for i in range(len(words) - 2):
            all_trigrams.append((words[i], words[i + 1], words[i + 2]))

    unique_1gram = len(set(all_words)) / max(len(all_words), 1)
    unique_2gram = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    unique_3gram = len(set(all_trigrams)) / max(len(all_trigrams), 1)

    # Combined score (weighted average)
    diversity_score = (unique_1gram * 0.3 + unique_2gram * 0.4 + unique_3gram * 0.3)

    return {
        "unique_1gram": unique_1gram,
        "unique_2gram": unique_2gram,
        "unique_3gram": unique_3gram,
        "diversity_score": diversity_score,
        "total_words": len(all_words),
    }


def compute_token_accuracy(
    model: Any,
    dataset: Any,
    num_samples: int = 100,
    top_k: int = 5,
) -> Dict[str, float]:
    """Compute next-token prediction accuracy.

    Measures how often the model's top predictions match the actual
    next token in the validation data.

    Args:
        model: Language model
        dataset: Dataset with tokenized text
        num_samples: Number of samples to evaluate
        top_k: Consider prediction correct if true token is in top-k

    Returns:
        Dictionary with:
        - top1_accuracy: Exact match accuracy
        - top5_accuracy: Top-5 accuracy
        - avg_rank: Average rank of correct token
    """
    model.eval()
    device = next(model.parameters()).device

    correct_top1 = 0
    correct_topk = 0
    total_ranks = []
    total = 0

    with torch.no_grad():
        for _ in range(num_samples):
            try:
                x, y = dataset.get_batch(batch_size=1, device=device)
                logits, _ = model(x)

                # Get last token prediction
                last_logits = logits[0, -1, :]  # [vocab_size]
                target = y[0, -1].item()

                # Top-1 accuracy
                pred = last_logits.argmax().item()
                if pred == target:
                    correct_top1 += 1

                # Top-k accuracy
                topk_preds = last_logits.topk(min(top_k, last_logits.size(0))).indices.tolist()
                if target in topk_preds:
                    correct_topk += 1

                # Compute rank of correct token
                sorted_indices = last_logits.argsort(descending=True).tolist()
                if target in sorted_indices:
                    rank = sorted_indices.index(target) + 1
                    total_ranks.append(rank)

                total += 1

            except Exception:
                continue

    return {
        "top1_accuracy": correct_top1 / max(total, 1),
        f"top{top_k}_accuracy": correct_topk / max(total, 1),
        "avg_rank": sum(total_ranks) / max(len(total_ranks), 1),
        "samples_evaluated": total,
    }


def compute_repetition_score(text: str, window_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
    """Detect repetitive patterns in generated text.

    Identifies degenerate repetition that indicates training issues
    or sampling problems.

    Args:
        text: Generated text to analyze
        window_sizes: Window sizes to check for repetition

    Returns:
        Dictionary with:
        - has_repetition: Whether significant repetition detected
        - repetition_ratio: Fraction of text that's repeated
        - longest_repeat: Length of longest repeated sequence
        - repeat_patterns: Most common repeated patterns
    """
    if window_sizes is None:
        window_sizes = [5, 10, 20]

    words = text.split()
    if len(words) < 10:
        return {
            "has_repetition": False,
            "repetition_ratio": 0.0,
            "longest_repeat": 0,
            "repeat_patterns": [],
        }

    # Find repeated n-grams
    repeated_positions = set()
    repeat_patterns = []

    for window_size in window_sizes:
        ngrams: Dict[Tuple[str, ...], List[int]] = {}
        for i in range(len(words) - window_size + 1):
            ngram = tuple(words[i:i + window_size])
            if ngram not in ngrams:
                ngrams[ngram] = []
            ngrams[ngram].append(i)

        # Find repeated ngrams
        for ngram, positions in ngrams.items():
            if len(positions) > 2:  # Repeated more than twice
                repeat_patterns.append({
                    "pattern": " ".join(ngram),
                    "count": len(positions),
                    "length": window_size,
                })
                for pos in positions:
                    for j in range(window_size):
                        repeated_positions.add(pos + j)

    repetition_ratio = len(repeated_positions) / max(len(words), 1)
    has_repetition = repetition_ratio > 0.3  # More than 30% repeated

    # Sort patterns by count
    repeat_patterns.sort(key=lambda x: x["count"], reverse=True)

    return {
        "has_repetition": has_repetition,
        "repetition_ratio": repetition_ratio,
        "longest_repeat": max((p["length"] for p in repeat_patterns), default=0),
        "repeat_patterns": repeat_patterns[:5],  # Top 5 patterns
    }
