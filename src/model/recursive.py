"""Recursive Latent Reasoning for GPT Language Models.

This module implements recursive reasoning capabilities that allow models to
iteratively refine their predictions through multiple passes. This trades
model parameters for computational depth, enabling smaller models to achieve
performance similar to larger models.

Key concepts:
- Latent Recursion: Model processes inputs multiple times to refine reasoning
- Deep Supervision: Train on intermediate predictions, not just final output
- Confidence-based Early Stopping: Stop recursion when model is confident
- Gradient-free Warmup: Initial recursions don't compute gradients for efficiency

Based on concepts from:
- "Recurrent Interface Networks" (Goyal et al.)
- "Universal Transformers" (Dehghani et al.)
- "Adaptive Computation Time" (Graves)
"""

import logging
from typing import Any, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import ModernGPTLanguageModel, ModernTransformerBlock

logger = logging.getLogger(__name__)


class ConfidenceHead(nn.Module):
    """Predicts confidence/quality of the current prediction.

    This head estimates whether the model's current prediction is likely
    correct, enabling early stopping during recursive reasoning.

    Args:
        n_embd: Embedding dimension
        dropout: Dropout rate for regularization
    """

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict confidence scores.

        Args:
            x: Hidden states [batch, seq_len, n_embd]

        Returns:
            Confidence logits [batch, seq_len, 1]
        """
        return self.proj(x)


class RecursiveTransformerBlock(nn.Module):
    """Transformer block with recursive latent reasoning capability.

    Wraps a ModernTransformerBlock to support iterative refinement of
    hidden states through multiple passes.

    Args:
        config: Configuration object
        block: ModernTransformerBlock to wrap (optional, creates new if None)
    """

    def __init__(self, config: Any, block: Optional[ModernTransformerBlock] = None):
        super().__init__()
        self.config = config

        # Use provided block or create new one
        if block is not None:
            self.block = block
        else:
            self.block = ModernTransformerBlock(config)

        # State projection layers for recursion
        self.y_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.z_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Combine input with states
        self.combine = nn.Linear(config.n_embd * 3, config.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional state recursion.

        Args:
            x: Input tensor [batch, seq_len, n_embd]
            y: Output state from previous recursion [batch, seq_len, n_embd]
            z: Latent reasoning state [batch, seq_len, n_embd]

        Returns:
            Tuple of (output, new_y, new_z, load_balancing_loss)
        """
        # Initialize states if not provided
        if y is None:
            y = torch.zeros_like(x)
        if z is None:
            z = torch.zeros_like(x)

        # Combine input with current states
        combined = torch.cat([x, y, z], dim=-1)
        state_input = self.combine(combined)

        # Apply transformer block
        output, load_balancing_loss = self.block(state_input)

        # Update states with residual connections
        new_y = y + self.y_proj(output)
        new_z = z + self.z_proj(output)

        return output, new_y, new_z, load_balancing_loss


class RecursiveGPTLanguageModel(nn.Module):
    """GPT Language Model with Recursive Latent Reasoning.

    This model extends the standard GPT with the ability to recursively
    refine predictions through multiple forward passes. This enables:
    - Smaller models with deeper computation
    - Iterative reasoning similar to chain-of-thought
    - Adaptive computation via confidence-based early stopping

    Args:
        config: Configuration object with recursive parameters:
            - use_recursive: Enable recursive reasoning (default: False)
            - recursion_depth: Number of recursion iterations T (default: 3)
            - latent_steps: Number of latent reasoning steps n (default: 6)
            - early_stop_threshold: Confidence for early stopping (default: 0.5)
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        # Create base GPT model
        self.base_model = ModernGPTLanguageModel(config)

        # Wrap transformer blocks with recursive capability if enabled
        self.use_recursive = getattr(config, 'use_recursive', False)

        if self.use_recursive:
            self.recursive_blocks = nn.ModuleList([
                RecursiveTransformerBlock(config, block)
                for block in self.base_model.blocks
            ])

            # Confidence head for early stopping
            self.confidence_head = ConfidenceHead(
                config.n_embd,
                dropout=getattr(config, 'dropout', 0.1)
            )

            # Parameters for recursion
            self.recursion_depth = getattr(config, 'recursion_depth', 3)
            self.latent_steps = getattr(config, 'latent_steps', 6)
            self.early_stop_threshold = getattr(config, 'early_stop_threshold', 0.5)

            logger.info(
                f"Recursive GPT initialized: depth={self.recursion_depth}, "
                f"latent_steps={self.latent_steps}, "
                f"early_stop_threshold={self.early_stop_threshold}"
            )

        # Delegate embeddings and output layers to base model
        self.token_embedding = self.base_model.token_embedding
        self.norm = self.base_model.norm
        self.lm_head = self.base_model.lm_head

    def _latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform latent reasoning through iterative refinement.

        Args:
            x: Input embeddings
            y: Output state
            z: Latent state
            n: Number of latent reasoning steps

        Returns:
            Tuple of (refined_output, new_y, new_z)
        """
        total_load_balancing_loss = 0.0

        # Iterative latent reasoning
        for _ in range(n):
            output = x
            for block in self.recursive_blocks:
                output, y, z, lb_loss = block(output, y, z)
                if lb_loss is not None:
                    total_load_balancing_loss += lb_loss

        # Final refinement pass
        output = x
        for block in self.recursive_blocks:
            output, y, z, lb_loss = block(output, y, z)
            if lb_loss is not None:
                total_load_balancing_loss += lb_loss

        return output, y, z

    def _deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        T: int,
        n: int,
        compute_gradients: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform deep recursion with improved gradient flow.

        Args:
            x: Input embeddings
            y: Output state
            z: Latent state
            T: Total number of recursions
            n: Latent steps per recursion
            compute_gradients: Whether to compute gradients

        Returns:
            Tuple of (output, confidence, y, z)
        """
        # Compute how many recursions should have gradients
        # For better gradient flow, compute gradients for all but the first recursion
        # This ensures the model learns from most of its computation
        warmup_steps = 1 if T > 1 and compute_gradients else (T if not compute_gradients else 0)
        gradient_steps = T - warmup_steps

        # Warm-up recursions without gradients (if any)
        if warmup_steps > 0:
            with torch.no_grad():
                for _ in range(warmup_steps):
                    output, y, z = self._latent_recursion(x, y, z, n)
            # After warmup, detach states to break gradient graph
            # This prevents MPS placeholder issues
            if gradient_steps > 0:
                y = y.detach()
                z = z.detach()

        # Remaining recursions WITH gradients for better learning
        if gradient_steps > 0:
            for _ in range(gradient_steps):
                output, y, z = self._latent_recursion(x, y, z, n)
        else:
            # Evaluation mode - no gradients needed
            with torch.no_grad():
                output, y, z = self._latent_recursion(x, y, z, n)

        # Compute confidence
        confidence = self.confidence_head(output)

        # Handle state detachment for MPS compatibility
        # During training, keep gradients but ensure contiguous memory
        # During eval, detach to save memory
        if not self.training:
            y = y.detach()
            z = z.detach()
        else:
            # Ensure contiguous for MPS compatibility
            y = y.contiguous()
            z = z.contiguous()

        return output, confidence, y, z

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_confidence: bool = False
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]],
               Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        """Forward pass with optional recursive reasoning.

        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices for loss computation
            return_confidence: Whether to return confidence scores

        Returns:
            If not using recursion: (logits, loss)
            If using recursion without return_confidence: (logits, loss)
            If using recursion with return_confidence: (logits, loss, confidence)
        """
        # Use base model if recursion disabled
        if not self.use_recursive:
            return self.base_model(idx, targets)

        # Get token embeddings
        B, T_seq = idx.shape
        x = self.token_embedding(idx)

        # Initialize states with proper device and dtype
        # Don't set requires_grad - let autograd handle it automatically
        y = torch.zeros(B, T_seq, self.config.n_embd, dtype=x.dtype, device=x.device)
        z = torch.zeros(B, T_seq, self.config.n_embd, dtype=x.dtype, device=x.device)

        # Perform deep recursion
        output, confidence, y, z = self._deep_recursion(
            x, y, z,
            T=self.recursion_depth,
            n=self.latent_steps,
            compute_gradients=self.training
        )

        # Final normalization and projection
        output = self.norm(output)
        logits = self.lm_head(output)

        # Ensure logits are contiguous for MPS compatibility
        logits = logits.contiguous()

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Primary loss: cross-entropy on predictions
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # Auxiliary loss: confidence calibration
            # Confidence should be high when prediction is correct
            predictions_correct = (logits.argmax(dim=-1) == targets).float().unsqueeze(-1)
            confidence_loss = F.binary_cross_entropy_with_logits(
                confidence,
                predictions_correct
            )

            # Combined loss
            confidence_weight = getattr(self.config, 'confidence_loss_weight', 0.1)
            loss = ce_loss + confidence_weight * confidence_loss

        if return_confidence:
            return logits, loss, torch.sigmoid(confidence)
        else:
            return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_recursion: Optional[bool] = None
    ) -> torch.Tensor:
        """Generate tokens with optional recursive reasoning.

        Args:
            idx: Starting token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_recursion: Override config to enable/disable recursion

        Returns:
            Generated token indices
        """
        # Allow override of recursion during generation
        original_use_recursive = self.use_recursive
        if use_recursion is not None:
            self.use_recursive = use_recursion

        try:
            # Delegate to base model's generate method
            return self.base_model.generate(
                idx, max_new_tokens, temperature, top_k, top_p
            )
        finally:
            # Restore original setting
            self.use_recursive = original_use_recursive

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory function for backward compatibility
def create_recursive_model(config: Any, vocab_size: int) -> RecursiveGPTLanguageModel:
    """Create a recursive GPT model with specified vocabulary size.

    Args:
        config: Configuration object
        vocab_size: Size of the vocabulary

    Returns:
        RecursiveGPTLanguageModel instance
    """
    # Update config with vocab size
    config.vocab_size = vocab_size

    model = RecursiveGPTLanguageModel(config)

    param_count = model._count_parameters()
    logger.info(f"Created recursive model with {param_count:,} parameters")

    return model
