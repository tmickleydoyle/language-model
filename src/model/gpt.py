"""Modern GPT Language Model Implementation.

This module implements a modernized GPT-style transformer with state-of-the-art
improvements including:
- RMSNorm instead of LayerNorm for better stability
- RoPE (Rotary Positional Encoding) for better positional understanding
- SwiGLU activation function for improved expressiveness
- Grouped Query Attention for efficiency
- Improved initialization schemes

The implementation follows recent advances in transformer architectures
while maintaining compatibility with the existing training pipeline.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable and efficient than LayerNorm, used in modern transformers
    like LLaMA and PaLM.
    
    Args:
        dim: Input dimension
        eps: Small value for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).
    
    Applies rotary position encoding to query and key vectors,
    providing better positional understanding and extrapolation
    to longer sequences than learned positional embeddings.
    
    Args:
        dim: Dimension of the embedding (should be head_dim)
        max_seq_len: Maximum sequence length
        base: Base for the geometric progression (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs_cis(max_seq_len)

    def _precompute_freqs_cis(self, seq_len: int) -> None:
        """Precompute cos and sin values for efficiency."""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Create complex exponentials
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary positional embedding.
        
        Args:
            x: Input tensor of shape (..., seq_len, dim)
            seq_len: Sequence length (if different from x.shape[-2])
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[-2]
            
        # Extend precomputed values if needed
        if seq_len > self.freqs_cis.shape[0]:
            self._precompute_freqs_cis(seq_len)
            
        # Get relevant frequencies
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Convert to complex for rotation
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        x_rotated = x_complex * freqs_cis.unsqueeze(0)
        
        # Convert back to real
        return torch.view_as_real(x_rotated).reshape(*x.shape)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention mechanism.
    
    Uses fewer key-value heads than query heads for efficiency while
    maintaining most of the representational power of full multi-head attention.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # Use fewer KV heads for efficiency (typically 1/4 to 1/8 of query heads)
        self.n_kv_head = getattr(config, 'n_kv_head', max(1, config.n_head // 4))
        self.n_rep = self.n_head // self.n_kv_head  # Repetition factor
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Mark output projection for special initialization
        self.o_proj._is_residual_projection = True
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(
            self.head_dim, 
            max_seq_len=config.block_size
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat key-value heads to match query heads."""
        batch_size, seq_len, n_kv_head, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return x.unsqueeze(3).expand(batch_size, seq_len, n_kv_head, self.n_rep, head_dim).reshape(
            batch_size, seq_len, n_kv_head * self.n_rep, head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for grouped query attention."""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        
        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)
        
        # Repeat K, V to match number of query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        
        # Compute attention with Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention (Flash Attention)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # Causal mask is handled automatically
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback to manual implementation
            out = self._manual_attention(q, k, v, seq_len)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.o_proj(out)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Manual attention computation fallback."""
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return attn_weights @ v


class SwiGLU(nn.Module):
    """SwiGLU activation function.
    
    Combines Swish activation with a gating mechanism for improved
    expressiveness in feedforward networks. Used in models like PaLM.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        
        # Calculate hidden dimension (typically 8/3 * n_embd for SwiGLU)
        hidden_dim = int(8 * config.n_embd / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        
        # Mark down projection for special initialization
        self.down_proj._is_residual_projection = True
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SwiGLU: Swish(gate) * up
        return self.dropout(self.down_proj(F.silu(gate) * up))


class ModernTransformerBlock(nn.Module):
    """Modern transformer block with RMSNorm, GQA, and SwiGLU.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block."""
        # Pre-norm attention with residual connection
        x = x + self.attention(self.attention_norm(x))
        
        # Pre-norm feedforward with residual connection
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x


class ModernGPTLanguageModel(nn.Module):
    """Modern GPT-style language model with state-of-the-art improvements.
    
    This model incorporates several modern techniques:
    - RMSNorm for better stability
    - RoPE for better positional encoding
    - Grouped Query Attention for efficiency
    - SwiGLU activation for better expressiveness
    - Improved initialization schemes
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        
        self._validate_config()
        self._initialize_embeddings()
        self._initialize_transformer_blocks()
        self._initialize_output_layers()
        self._tie_weights()
        self._initialize_weights()
        
        logger.info(f"Modern GPT model initialized with {self._count_parameters():,} parameters")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.config.n_embd % self.config.n_head != 0:
            raise ValueError(
                f"n_embd ({self.config.n_embd}) must be divisible by n_head ({self.config.n_head})"
            )

    def _initialize_embeddings(self) -> None:
        """Initialize token embeddings (no positional embeddings with RoPE)."""
        self.token_embedding = nn.Embedding(self.vocab_size, self.config.n_embd)

    def _initialize_transformer_blocks(self) -> None:
        """Initialize modern transformer blocks."""
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(self.config) for _ in range(self.config.n_layer)
        ])

    def _initialize_output_layers(self) -> None:
        """Initialize final norm and language modeling head."""
        self.norm = RMSNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.vocab_size, bias=False)

    def _tie_weights(self) -> None:
        """Tie weights between token embedding and language modeling head."""
        self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self) -> None:
        """Initialize weights with improved schemes."""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with modern schemes."""
        if isinstance(module, nn.Linear):
            # Use different initialization for different types of layers
            if hasattr(module, '_is_residual_projection'):
                # Smaller initialization for residual projections
                std = 0.02 / (2 * self.config.n_layer) ** 0.5
            else:
                # Standard initialization for other linear layers
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model."""
        batch_size, seq_len = idx.shape
        
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length ({seq_len}) exceeds block_size ({self.config.block_size})")
        
        # Token embeddings only (RoPE handles positional information)
        x = self.token_embedding(idx)
        
        # Apply transformer blocks with mixed precision
        device_type = "cuda" if x.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type, enabled=getattr(self.config, 'fp16', False) and x.device.type == "cuda"):
            for block in self.blocks:
                x = block(x)
            
            # Final normalization and projection
            x = self.norm(x)
            logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss

    def generate(
        self,
        idx: Union[torch.Tensor, List[List[int]]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate new tokens with improved sampling options."""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        
        # Prepare input
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=next(self.parameters()).device)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if too long
                idx_cond = idx[:, -self.config.block_size:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_logits, _ = torch.topk(logits, top_k)
                    logits[logits < top_logits[:, [-1]]] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
        
        return idx

    def get_num_params(self) -> Dict[str, int]:
        """Get detailed parameter count information."""
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

    def configure_optimizers(self, learning_rate: float, weight_decay: float, device_type: str):
        """Configure optimizers with parameter grouping."""
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) >= 2:  # Weight matrices
                    decay_params.append(param)
                else:  # Biases and layer norm parameters
                    no_decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Use AdamW with better hyperparameters for modern transformers
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=device_type == "cuda"  # Use fused optimizer on CUDA
        )
        
        return optimizer


def create_modern_model(config: Any, vocab_size: int) -> ModernGPTLanguageModel:
    """Create a modern GPT model with the given configuration."""
    config.vocab_size = vocab_size
    
    # Add default values for new config parameters if not present
    if not hasattr(config, 'n_kv_head'):
        config.n_kv_head = max(1, config.n_head // 4)  # Use 1/4 of query heads for KV
    
    model = ModernGPTLanguageModel(config)
    model = model.to(config.device)
    
    logger.info(f"Created modern model with {model._count_parameters():,} parameters")
    logger.info(f"Query heads: {config.n_head}, KV heads: {config.n_kv_head}")
    
    return model


# Backward compatibility
GPTLanguageModel = ModernGPTLanguageModel
create_model = create_modern_model
