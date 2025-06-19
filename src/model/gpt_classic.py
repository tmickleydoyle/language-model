"""Classic GPT Language Model Implementation.

This module implements a traditional GPT-style transformer architecture
with standard components including:
- Multi-Head Self-Attention
- Feed-Forward Networks with SiLU activation
- Layer Normalization
- Positional Embeddings
- Standard transformer blocks

This implementation provides a baseline and compatibility layer
for existing code while the modern implementation explores newer techniques.
"""

import inspect
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism.
    
    Standard self-attention implementation with causal masking for
    autoregressive language modeling.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        # Query, Key, Value projections for all heads in batch
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure that attention is only applied to left in sequence
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.size()
        
        # Calculate query, key, values for all heads in batch
        qkv = self.qkv(x)
        q, k, v = qkv.split(n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v
        
        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        y = self.proj(y)
        
        return y


class FeedForward(nn.Module):
    """Feed-Forward Network with SiLU activation.
    
    Standard MLP with expansion factor of 4x and SiLU activation.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.SiLU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Classic transformer block with LayerNorm and post-normalization.
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.ffn_norm = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Post-norm attention with residual connection
        x = x + self.attention_norm(self.attention(x))
        
        # Post-norm feedforward with residual connection
        x = x + self.ffn_norm(self.feed_forward(x))
        
        return x


class GPTLanguageModel(nn.Module):
    """Classic GPT language model implementation.
    
    Standard GPT architecture with:
    - Token and positional embeddings
    - Stack of transformer blocks
    - Layer normalization
    - Language modeling head
    
    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 initialization scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for loss computation
            
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: (logits, loss)
        """
        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm and language modeling head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """Generate text by sampling from the model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling
            
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get the predictions
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    @torch.no_grad()
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # First estimate the number of flops we do per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 peak flops
        flops_achieved = flops_per_iter / dt # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay: float, learning_rate: float, 
                           betas: Tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        """Configure optimizer with weight decay."""
        # Start with all candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer


def create_model(config: Any, vocab_size: int) -> GPTLanguageModel:
    """Create a classic GPT language model.
    
    Args:
        config: Configuration object containing model hyperparameters
        vocab_size: Size of the vocabulary
        
    Returns:
        GPTLanguageModel instance
    """
    # Update config with vocab_size
    config.vocab_size = vocab_size
    
    model = GPTLanguageModel(config)
    logger.info(f"Created classic GPT model with {model.get_num_params():,} parameters")
    
    return model
