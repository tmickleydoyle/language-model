"""GPT Language Model Implementation.

This module implements a GPT-style transformer language model with multi-head
self-attention, feed-forward networks, and autoregressive generation capabilities.

The implementation includes:
- MultiHeadAttention with causal masking and mixed precision support
- FeedForward network with configurable hidden dimension
- TransformerBlock combining attention and feed-forward layers
- GPTLanguageModel with token/positional embeddings and generation capabilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module with causal masking.

    This implementation uses fused QKV computation for efficiency and applies
    causal masking to prevent attention to future tokens in the sequence.

    Args:
        config: Configuration object containing model hyperparameters

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        qkv: Fused linear layer for Q, K, V computation
        proj: Output projection layer
        dropout: Dropout layer
        causal_mask: Registered buffer for causal masking
    """

    def __init__(self, config: Any) -> None:
        """Initialize the multi-head attention module."""
        super().__init__()

        self._validate_config(config)

        self.config = config
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Fused linear layer for Q, K, V computation
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout layers
        self.dropout = nn.Dropout(config.dropout)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def _validate_config(self, config: Any) -> None:
        """Validate configuration parameters."""
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding dimension ({config.n_embd}) must be divisible by "
                f"number of heads ({config.n_head})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Use mixed precision if enabled
        with torch.amp.autocast("cuda", enabled=self.config.fp16):
            # Compute and split Q, K, V in one pass
            qkv_output = self._compute_qkv(x, batch_size, seq_len)
            q, k, v = self._split_qkv(qkv_output)

            # Compute attention
            attention_output = self._compute_attention(q, k, v, seq_len)

            # Final projection and dropout
            output = self.dropout(self.proj(attention_output))

        return output  # type: ignore[no-any-return]

    def _compute_qkv(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Compute Q, K, V tensors."""
        qkv = self.qkv(x)  # (B, T, 3*C)
        return qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)

    def _split_qkv(self, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor and transpose for attention computation."""
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, num_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, T, head_dim)

        return q, k, v

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Compute scaled dot-product attention with causal masking."""
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attention_scores = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, T, T)

        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]  # type: ignore
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        out = attention_probs @ v  # (B, num_heads, T, head_dim)

        # Reshape back to original dimensions
        batch_size = q.size(0)
        embed_dim = self.num_heads * self.head_dim
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)


class FeedForward(nn.Module):
    """Feed-forward network used within transformer blocks.

    Implements a two-layer MLP with SiLU activation and dropout.
    The hidden dimension is typically 4x the embedding dimension.

    Args:
        config: Configuration object containing model hyperparameters

    Attributes:
        net: Sequential network containing linear layers, activation, and dropout
    """

    def __init__(self, config: Any) -> None:
        """Initialize the feed-forward network."""
        super().__init__()

        hidden_dim = self._calculate_hidden_dim(config.n_embd)

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def _calculate_hidden_dim(self, n_embd: int) -> int:
        """Calculate hidden dimension (typically 4x embedding dimension)."""
        return 4 * n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape as input
        """
        return self.net(x)  # type: ignore[no-any-return]


class TransformerBlock(nn.Module):
    """Transformer block combining self-attention and feed-forward layers.

    Implements the standard transformer architecture with pre-normalization
    and residual connections.

    Args:
        config: Configuration object containing model hyperparameters

    Attributes:
        attention: Multi-head attention layer
        feed_forward: Feed-forward network
        ln1: Layer normalization before attention
        ln2: Layer normalization before feed-forward
    """

    def __init__(self, config: Any) -> None:
        """Initialize the transformer block."""
        super().__init__()

        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection and pre-normalization
        x = x + self.attention(self.ln1(x))

        # Feed-forward with residual connection and pre-normalization
        x = x + self.feed_forward(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    """GPT-style language model with transformer architecture.

    This model implements a decoder-only transformer with token and positional
    embeddings, multiple transformer blocks, and a language modeling head.

    Args:
        config: Configuration object containing model hyperparameters
                including vocab_size

    Attributes:
        config: Model configuration
        vocab_size: Vocabulary size
        token_embedding: Token embedding layer
        position_embedding: Positional embedding layer
        blocks: List of transformer blocks
        ln_f: Final layer normalization
        lm_head: Language modeling head
    """

    def __init__(self, config: Any) -> None:
        """Initialize the GPT language model."""
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self._validate_vocab_size()
        self._initialize_embeddings()
        self._initialize_transformer_blocks()
        self._initialize_output_layers()
        self._initialize_weights()

        logger.info(
            f"GPT model initialized with {self._count_parameters():,} parameters"
        )

    def _validate_vocab_size(self) -> None:
        """Validate vocabulary size."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    def _initialize_embeddings(self) -> None:
        """Initialize token and positional embeddings."""
        self.token_embedding = nn.Embedding(self.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.n_embd)

        # Register position indices buffer
        self.register_buffer(
            "position_ids", torch.arange(self.config.block_size).unsqueeze(0)
        )

    def _initialize_transformer_blocks(self) -> None:
        """Initialize transformer blocks."""
        self.blocks = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layer)
        ])

    def _initialize_output_layers(self) -> None:
        """Initialize final layer norm and language modeling head."""
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.vocab_size)

    def _initialize_weights(self) -> None:
        """Initialize weights using normal distribution."""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for loss computation (optional)

        Returns:
            Tuple of (logits, loss) where:
            - logits: Output logits of shape (batch_size, seq_len, vocab_size)
            - loss: Cross-entropy loss if targets provided, else None

        Raises:
            ValueError: If sequence length exceeds block_size
        """
        batch_size, seq_len = idx.shape

        self._validate_sequence_length(seq_len)

        # Compute embeddings and apply transformer blocks
        x = self._compute_embeddings(idx, seq_len)
        logits = self._apply_transformer_blocks(x)

        # Compute loss if targets provided
        loss = self._compute_loss(logits, targets) if targets is not None else None

        return logits, loss

    def _validate_sequence_length(self, seq_len: int) -> None:
        """Validate that sequence length doesn't exceed block_size."""
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds "
                f"block_size ({self.config.block_size})"
            )

    def _compute_embeddings(self, idx: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute token and positional embeddings."""
        token_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(
            self.position_ids[:, :seq_len]  # type: ignore
        )  # (1, T, C)
        return token_emb + pos_emb

    def _apply_transformer_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer blocks and final processing."""
        with torch.amp.autocast("cuda", enabled=self.config.fp16):
            for block in self.blocks:
                x = block(x)

            # Final layer norm and projection
            x = self.ln_f(x)
            logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        # Flatten for cross-entropy computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        return F.cross_entropy(logits_flat, targets_flat)

    def generate(
        self,
        idx: Union[torch.Tensor, List[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        Args:
            idx: Initial token indices, shape (batch_size, seq_len) or list of lists
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If specified, only sample from top-k most likely tokens

        Returns:
            Generated sequence including initial tokens

        Raises:
            ValueError: If max_new_tokens is not positive
        """
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

        # Prepare input tensor
        idx = self._prepare_generation_input(idx)

        # Set to evaluation mode
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if too long
                idx_cond = idx[:, -self.config.block_size :]

                # Forward pass
                logits, _ = self(idx_cond)

                # Process logits for next token prediction
                next_token = self._sample_next_token(logits, temperature, top_k)

                # Append to sequence
                idx = torch.cat([idx, next_token], dim=1)

        return idx

    def _prepare_generation_input(
        self, idx: Union[torch.Tensor, List[List[int]]]
    ) -> torch.Tensor:
        """Prepare input tensor for generation."""
        # Convert to tensor if needed
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=self.config.device)

        # Ensure proper shape
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        return idx

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: float, top_k: Optional[int]
    ) -> torch.Tensor:
        """Sample next token from logits."""
        # Get logits for last token only
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # Apply top-k filtering if specified
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, top_k)
            logits[logits < top_logits[:, [-1]]] = float("-inf")

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def get_num_params(self) -> Dict[str, int]:
        """Get detailed parameter count information.

        Returns:
            Dictionary with parameter counts for each module and total
        """
        param_counts = {}
        total_params = 0

        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    param_counts[name] = num_params
                    total_params += num_params

        param_counts["total"] = total_params
        return param_counts


def create_model(config: Any, vocab_size: int) -> GPTLanguageModel:
    """Create a GPT model with the given configuration.

    Args:
        config: Configuration object
        vocab_size: Vocabulary size

    Returns:
        Initialized GPT model on the specified device
    """
    # Update config with vocab_size
    config.vocab_size = vocab_size
    model = GPTLanguageModel(config)
    model = model.to(config.device)

    logger.info(f"Created model with {model._count_parameters():,} parameters")
    logger.info(f"Model device: {config.device}")

    return model


if __name__ == '__main__':
    # This section is for testing purposes only
    from ..config import Config

    # Initialize configuration and model
    config = Config()
    vocab_size = 5000

    model = create_model(config, vocab_size)

    # Test forward pass
    batch_size, seq_len = 2, 32
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=config.device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=config.device)

    model.train()
    logits, loss = model(idx, targets)
    logger.info(f"Test forward pass completed. Loss: {loss.item():.4f}")

    # Test generation
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 10), device=config.device)
    generated = model.generate(prompt, max_new_tokens=20)
    logger.info(f"Test generation completed. Shape: {generated.shape}")
