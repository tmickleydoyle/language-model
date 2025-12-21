"""Benchmark tests for model quality assessment.

This module provides benchmarks for evaluating language model quality:
- Syntax validation (for code models)
- Completion benchmarks with standard prompts
- Repetition analysis
- Comprehensive quality evaluation
"""

import ast
import logging
from typing import Any, Dict, List, Optional

from .metrics import (
    compute_perplexity,
    compute_diversity_score,
    compute_repetition_score,
)

logger = logging.getLogger(__name__)


def run_syntax_validation(samples: List[str]) -> Dict[str, Any]:
    """Validate Python syntax in generated code samples.

    Tests whether generated code is syntactically valid Python.
    Useful for evaluating code-focused models.

    Args:
        samples: List of generated code samples

    Returns:
        Dictionary with:
        - valid_count: Number of syntactically valid samples
        - valid_ratio: Percentage of valid samples
        - errors: List of syntax errors encountered
    """
    valid_count = 0
    errors = []

    for i, sample in enumerate(samples):
        try:
            # Try to parse as Python code
            ast.parse(sample)
            valid_count += 1
        except SyntaxError as e:
            errors.append({
                "sample_index": i,
                "error": str(e),
                "line": getattr(e, "lineno", None),
            })

    return {
        "valid_count": valid_count,
        "total_count": len(samples),
        "valid_ratio": valid_count / max(len(samples), 1),
        "sample_errors": errors[:10],  # First 10 errors
    }


def run_completion_benchmark(
    model: Any,
    tokenizer: Any,
    prompts: Optional[List[str]] = None,
    num_tokens: int = 50,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Run completion benchmark with standard prompts.

    Generates completions for a set of prompts and evaluates
    the quality of the generated text.

    Args:
        model: Language model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompts to complete
        num_tokens: Tokens to generate per prompt
        temperature: Sampling temperature

    Returns:
        Dictionary with:
        - completions: List of generated completions
        - diversity: Diversity metrics
        - avg_length: Average completion length
    """
    if prompts is None:
        prompts = [
            "def fibonacci(",
            "class Database:",
            "# Calculate the sum of",
            "The function returns",
            "import ",
        ]

    import torch

    model.eval()
    device = next(model.parameters()).device

    completions = []

    with torch.no_grad():
        for prompt in prompts:
            try:
                # Encode prompt
                encoded = tokenizer.encode(prompt)
                input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

                # Generate
                if hasattr(model, "generate"):
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=num_tokens,
                        temperature=temperature,
                        top_k=50,
                    )
                else:
                    # Fallback: simple autoregressive generation
                    output_ids = input_ids.clone()
                    for _ in range(num_tokens):
                        logits, _ = model(output_ids)
                        next_logits = logits[:, -1, :] / temperature
                        probs = torch.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        output_ids = torch.cat([output_ids, next_token], dim=1)

                # Decode
                generated = tokenizer.decode(output_ids[0].tolist())
                completions.append({
                    "prompt": prompt,
                    "completion": generated,
                    "length": len(generated.split()),
                })

            except Exception as e:
                logger.warning(f"Completion failed for '{prompt}': {e}")
                completions.append({
                    "prompt": prompt,
                    "completion": f"[Error: {e}]",
                    "length": 0,
                })

    # Compute metrics
    texts = [c["completion"] for c in completions]
    diversity = compute_diversity_score(texts)
    avg_length = sum(c["length"] for c in completions) / max(len(completions), 1)

    return {
        "completions": completions,
        "diversity": diversity,
        "avg_length": avg_length,
        "num_prompts": len(prompts),
    }


def run_repetition_analysis(samples: List[str]) -> Dict[str, Any]:
    """Analyze repetition patterns in generated text.

    Detects degenerate repetition that indicates training problems.

    Args:
        samples: List of generated text samples

    Returns:
        Dictionary with:
        - samples_with_repetition: Count of samples with significant repetition
        - avg_repetition_ratio: Average repetition across samples
        - worst_samples: Indices of most repetitive samples
    """
    results = []

    for i, sample in enumerate(samples):
        rep_score = compute_repetition_score(sample)
        results.append({
            "index": i,
            "has_repetition": rep_score["has_repetition"],
            "repetition_ratio": rep_score["repetition_ratio"],
        })

    # Summary statistics
    samples_with_rep = sum(1 for r in results if r["has_repetition"])
    avg_ratio = sum(r["repetition_ratio"] for r in results) / max(len(results), 1)

    # Find worst samples
    results.sort(key=lambda x: x["repetition_ratio"], reverse=True)
    worst_indices = [r["index"] for r in results[:5] if r["has_repetition"]]

    return {
        "samples_with_repetition": samples_with_rep,
        "total_samples": len(samples),
        "repetition_rate": samples_with_rep / max(len(samples), 1),
        "avg_repetition_ratio": avg_ratio,
        "worst_sample_indices": worst_indices,
    }


def evaluate_model_quality(
    model: Any,
    tokenizer: Any,
    val_dataset: Any,
    num_samples: int = 20,
    prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Comprehensive model quality evaluation.

    Runs multiple benchmarks and returns a summary of model quality.

    Args:
        model: Language model
        tokenizer: Tokenizer
        val_dataset: Validation dataset
        num_samples: Number of samples to generate
        prompts: Custom prompts for generation

    Returns:
        Dictionary with comprehensive quality metrics:
        - perplexity: Validation perplexity
        - diversity: Diversity score
        - repetition_rate: Rate of repetitive samples
        - syntax_valid_rate: Rate of syntactically valid code (if applicable)
        - quality_score: Combined quality score (0-100)
    """
    import torch

    logger.info("Running comprehensive model evaluation...")

    # 1. Compute perplexity on validation set
    model.eval()
    device = next(model.parameters()).device
    losses = []

    with torch.no_grad():
        for _ in range(min(100, len(val_dataset))):
            try:
                x, y = val_dataset.get_batch(batch_size=1, device=device)
                _, loss = model(x, y)
                losses.append(loss.item())
            except Exception:
                continue

    avg_loss = sum(losses) / max(len(losses), 1)
    perplexity = compute_perplexity(avg_loss)

    # 2. Generate samples for analysis
    completion_results = run_completion_benchmark(
        model, tokenizer, prompts, num_tokens=50
    )

    samples = [c["completion"] for c in completion_results["completions"]]

    # 3. Diversity analysis
    diversity = compute_diversity_score(samples)

    # 4. Repetition analysis
    repetition = run_repetition_analysis(samples)

    # 5. Syntax validation (for code models)
    syntax = run_syntax_validation(samples)

    # 6. Compute overall quality score (0-100)
    # Lower perplexity = better (map 10-100 to 100-0)
    ppl_score = max(0, min(100, 100 - (perplexity - 10) * 1.5))

    # Higher diversity = better
    div_score = diversity["diversity_score"] * 100

    # Lower repetition = better
    rep_score = (1 - repetition["repetition_rate"]) * 100

    # Weighted combination
    quality_score = (ppl_score * 0.5 + div_score * 0.3 + rep_score * 0.2)

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "diversity": diversity,
        "diversity_score": diversity["diversity_score"],
        "repetition": repetition,
        "repetition_rate": repetition["repetition_rate"],
        "syntax": syntax,
        "syntax_valid_rate": syntax["valid_ratio"],
        "quality_score": quality_score,
        "samples_evaluated": len(losses),
        "samples_generated": len(samples),
    }
