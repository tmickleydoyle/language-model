import torch

class Config:
    # Model hyperparameters
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 128   # maximum context length for predictions
    n_embd = 384    # embedding dimension
    n_head = 6      # number of attention heads
    n_layer = 6     # number of transformer blocks
    dropout = 0.2   # dropout rate
    vocab_size = 512  # Added this parameter for BPE tokenizer
    
    # Training hyperparameters
    max_iters = 2500
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    
    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 1337

    def __init__(self):
        torch.manual_seed(self.seed) 