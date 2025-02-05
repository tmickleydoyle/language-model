import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Define your configuration object (or use your own)
class Config:
    def __init__(self):
        self.n_embd = 256       # embedding dimension
        self.n_head = 8         # number of heads
        self.n_layer = 6        # number of transformer blocks
        self.block_size = 128   # context window
        self.dropout = 0.1      # dropout rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = False       # set to True to enable mixed precision (if supported)

config = Config()

###############################################################################
# MultiHeadAttention with fused QKV and causal masking (cached as a buffer)
###############################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # Fused linear layer to project into query, key, and value
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # Cache the causal mask
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x):
        B, T, C = x.shape

        # Optionally use mixed precision for the forward pass
        context = autocast() if config.fp16 else torch.enable_grad()
        with context:
            # Compute Q, K, V in one shot and reshape
            qkv = self.qkv(x)  # (B, T, 3 * C)
            # Reshape to (B, T, num_heads, 3 * head_dim)
            qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)
            # Split along last dimension to get Q, K, V each with shape (B, T, num_heads, head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            
            # Transpose for multi-head attention: (B, num_heads, T, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, T, T)
            # Apply the causal mask. Use only the first T tokens of the cached mask.
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)

            # Weighted sum of the values.
            out = att @ v  # (B, num_heads, T, head_dim)
            # Transpose and reshape back to (B, T, C)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.dropout(self.proj(out))
        return out

###############################################################################
# Feed Forward Network
###############################################################################
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

###############################################################################
# Transformer Block
###############################################################################
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

###############################################################################
# GPT Language Model
###############################################################################
class GPTLanguageModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        # Token and position embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # Cache positional indices to avoid recomputation on each forward.
        self.register_buffer('pos_idx', torch.arange(config.block_size).unsqueeze(0))

        # Transformer blocks.
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
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
        idx: (B, T) input token indices.
        targets: (B, T) target token indices.
        """
        B, T = idx.shape
        # Token embeddings: (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        # Use the pre-cached position indices (trimmed to T)
        pos_emb = self.position_embedding_table(self.pos_idx[:, :T])
        x = tok_emb + pos_emb

        # Optionally use AMP for the transformer blocks
        context = autocast() if config.fp16 else torch.enable_grad()
        with context:
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            if targets is not None:
                # Reshape logits and targets for the cross entropy loss.
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate new tokens.
        idx: (B, T) input token indices.
        max_new_tokens: number of tokens to generate.
        """
        # Switch to evaluation mode and disable dropout
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop the context if needed
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                # Focus on the last time step.
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # Sample the next token
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

###############################################################################
# Example usage and optional TorchScript compilation
###############################################################################
if __name__ == '__main__':
    # Example parameters.
    vocab_size = 5000
    model = GPTLanguageModel(config, vocab_size).to(config.device)

    # Optionally compile with TorchScript (if your model is compatible)
    try:
        model = torch.jit.script(model)
        print("TorchScript compilation succeeded.")
    except Exception as e:
        print("TorchScript compilation failed:", e)

    # Dummy input
    B, T = 4, 64  # batch size and sequence length
    idx = torch.randint(0, vocab_size, (B, T), device=config.device)
    targets = torch.randint(0, vocab_size, (B, T), device=config.device)

    # Training forward pass
    model.train()
    logits, loss = model(idx, targets)
    print("Training loss:", loss.item())

    # Inference generation example
    prompt = torch.randint(0, vocab_size, (B, T), device=config.device)
    generated = model.generate(prompt, max_new_tokens=20)
    print("Generated shape:", generated.shape)
