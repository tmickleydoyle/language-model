import torch

class Config:
    """
    Configuration class to store hyperparameters and other settings.

    Attributes:
        batch_size: Number of independent sequences to process in parallel.
        block_size: Maximum context length for predictions.
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
        vocab_size: Size of the vocabulary.
        max_iters: Maximum number of training iterations.
        eval_interval: Number of iterations between evaluation runs.
        learning_rate: Learning rate for the optimizer.
        eval_iters: Number of iterations to run evaluation for.
        device: Device to run the model on.
        seed: Random seed for reproducibility.
    """
    # Model hyperparameters
    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 256   # maximum context length for predictions
    n_embd = 384    # embedding dimension
    n_head = 4      # number of attention heads
    n_layer = 4     # number of transformer blocks
    dropout = 0.05   # dropout rate
    vocab_size = 2500  # Added this parameter for BPE tokenizer

    # Training hyperparameters
    max_iters = 2500
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 100

    # System
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 1337

    def __init__(self):
        torch.manual_seed(self.seed)
