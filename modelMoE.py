import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

###############################################################################
# Configuration
###############################################################################
class Config:
    def __init__(self):
        self.n_embd = 256         # Embedding dimension.
        self.n_head = 8           # Number of attention heads.
        self.n_layer = 6          # Number of transformer blocks.
        self.block_size = 128     # Maximum sequence length.
        self.dropout = 0.1        # Dropout probability.
        self.n_experts = 4        # Number of experts in the MoE.
        self.moe_loss_coef = 0.01 # Coefficient for auxiliary load-balancing loss.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = False         # Enable mixed precision if True.

config = Config()

###############################################################################
# MultiHeadAttention Module (unchanged)
###############################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # Fused linear layer to compute Q, K, V.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # Cache the causal mask.
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        context = autocast() if config.fp16 else torch.enable_grad()
        with context:
            qkv = self.qkv(x)  # (B, T, 3 * C)
            qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)  # Each is (B, T, num_heads, head_dim)
            # Transpose to shape (B, num_heads, T, head_dim)
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            # Scaled dot-product attention.
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            # Weighted sum of values.
            out = att @ v  # (B, num_heads, T, head_dim)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.dropout(self.proj(out))
        return out

###############################################################################
# Sparse (Top-1) Mixture-of-Experts Feed-Forward Module
###############################################################################
class SparseMoEFeedForward(nn.Module):
    """
    A feed-forward network that routes each token to a single expert (top-1 gating).
    In addition to returning the transformed output, it also computes an auxiliary
    load-balancing loss.
    """
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        # Build a list of expert feed-forward networks.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.ReLU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout),
            )
            for _ in range(self.n_experts)
        ])
        # Gating network produces a logit for each expert.
        self.gate = nn.Linear(config.n_embd, self.n_experts)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            output: Tensor of shape (B, T, C), the combined output from experts.
            aux_loss: A scalar tensor for the auxiliary load-balancing loss.
        """
        B, T, C = x.shape
        # Compute gate logits and softmax to obtain probabilities.
        gate_logits = self.gate(x)  # (B, T, n_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)  # (B, T, n_experts)
        # For each token, select the expert with the highest score.
        top_gate_scores, top_expert_idx = gate_scores.max(dim=-1)  # (B, T)

        # --- Auxiliary Load-Balancing Loss Computation ---
        # "Importance" is the soft token count for each expert.
        importance = gate_scores.sum(dim=(0, 1))  # (n_experts,)
        # "Load" is the hard token count from the top-1 routing.
        load = torch.zeros(self.n_experts, device=x.device)
        for expert in range(self.n_experts):
            load[expert] = (top_expert_idx == expert).sum()
        total_tokens = B * T
        importance_norm = importance / total_tokens  # Normalize importance.
        load_norm = load / total_tokens              # Normalize load.
        # The auxiliary loss encourages load balancing across experts.
        aux_loss = self.n_experts * (importance_norm * load_norm).sum()

        # --- Routing: Process tokens for each expert ---
        # Initialize an output tensor.
        output = torch.zeros_like(x)
        # Loop over experts. (For a small number of experts, a loop is acceptable.)
        for expert in range(self.n_experts):
            # Create a boolean mask of tokens assigned to the current expert.
            mask = (top_expert_idx == expert)  # Shape: (B, T)
            if mask.sum() == 0:
                continue  # Skip if no token is assigned.
            # Extract the tokens assigned to this expert.
            expert_input = x[mask]  # (n_tokens_expert, C)
            # Process with the expert network.
            expert_output = self.experts[expert](expert_input)  # (n_tokens_expert, C)
            # Multiply the output by the gating probability.
            expert_gate = top_gate_scores[mask].unsqueeze(-1)  # (n_tokens_expert, 1)
            expert_output = expert_output * expert_gate
            # Scatter the expert outputs back to the appropriate locations.
            output[mask] = expert_output

        return output, aux_loss

###############################################################################
# Transformer Block with Sparse MoE Feed-Forward
###############################################################################
class Block(nn.Module):
    """
    A single transformer block that uses multi-head self-attention followed
    by a sparse MoE feed-forward network. Each block returns both its output
    and the auxiliary loss from its MoE layer.
    """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = SparseMoEFeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Attention with residual connection.
        x = x + self.sa(self.ln1(x))
        # MoE feed-forward with residual connection.
        ffwd_out, aux_loss = self.ffwd(self.ln2(x))
        x = x + ffwd_out
        return x, aux_loss

###############################################################################
# GPT Language Model with MoE Transformer Blocks
###############################################################################
class GPTLanguageModelMoE(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        # Token and position embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.register_buffer('pos_idx', torch.arange(config.block_size).unsqueeze(0))

        # Instead of nn.Sequential, use a ModuleList so that each block’s
        # auxiliary loss can be collected.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Tensor of shape (B, T) with token indices.
            targets: (Optional) Tensor of shape (B, T) with target token indices.
        Returns:
            logits: The output logits of shape (B, T, vocab_size).
            loss: The total loss (cross-entropy plus weighted auxiliary MoE loss) if targets is provided.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                   # (B, T, C)
        pos_emb = self.position_embedding_table(self.pos_idx[:, :T])    # (1, T, C)
        x = tok_emb + pos_emb

        # Accumulate auxiliary loss from each transformer block.
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            ce_loss = F.cross_entropy(logits_flat, targets_flat)
            # Combine the main cross-entropy loss with the MoE auxiliary loss.
            loss = ce_loss + self.config.moe_loss_coef * total_aux_loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate new tokens.
        Args:
            idx: Tensor or list of shape (B, T) of input token indices.
            max_new_tokens: Number of tokens to generate.
        Returns:
            idx: Tensor of shape (B, T + max_new_tokens) with generated tokens.
        """
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx, dtype=torch.long, device=self.config.device)
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop the context if needed.
                idx_cond = idx[:, -self.config.block_size:]
                # In generation, we ignore the auxiliary loss.
                x = self.token_embedding_table(idx_cond) + \
                    self.position_embedding_table(self.pos_idx[:, :idx_cond.size(1)])
                total_aux_loss = 0.0  # Dummy value.
                for block in self.blocks:
                    x, aux_loss = block(x)
                    total_aux_loss += aux_loss
                x = self.ln_f(x)
                logits = self.lm_head(x)
                logits = logits[:, -1, :]  # Focus on the last time step.
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

###############################################################################
# Example Usage
###############################################################################
if __name__ == '__main__':
    vocab_size = 5000
    model = GPTLanguageModelMoE(config, vocab_size).to(config.device)

    # (Optionally) compile with TorchScript if desired.
    try:
        model = torch.jit.script(model)
        print("TorchScript compilation succeeded.")
    except Exception as e:
        print("TorchScript compilation failed:", e)

    # Dummy training example.
    B, T = 4, 64  # Batch size and sequence length.
    idx = torch.randint(0, vocab_size, (B, T), device=config.device)
    targets = torch.randint(0, vocab_size, (B, T), device=config.device)

    model.train()
    logits, loss = model(idx, targets)
    print("Training loss:", loss.item())

    # Inference generation example.
    prompt = [list(torch.randint(0, vocab_size, (T,)).tolist()) for _ in range(B)]
    generated = model.generate(prompt, max_new_tokens=20)
    print("Generated shape:", generated.shape)
