import torch
from typing import Dict, List, Tuple
from bpe import BPE

class TextDataset:
    """
    TextDataset class for loading and encoding text data.

    Args:
        config: Config object containing hyperparameters and device information.

    Attributes:
        config: Config object containing hyperparameters and device information.
        chars: List of characters in the vocabulary.
        vocab_size: Size of the vocabulary.
        stoi: Dictionary mapping characters to indices.
        itos: Dictionary mapping indices to characters.
        train_data: Tensor containing training data.
        val_data: Tensor containing validation data.
        bpe: Byte Pair Encoding object
    """
    def __init__(self, config):
        self.config = config
        self.chars: List[str] = []
        self.vocab_size: int = 0
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.train_data: torch.Tensor = None
        self.val_data: torch.Tensor = None
        self.bpe = BPE()

    def load_data(self, filepath: str):
        """Load text data from file and encode using BPE

        Args:
            filepath: Path to text file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Train BPE on the text
        self.bpe.train(
            text=text,
            max_vocab_size=self.config.vocab_size,
            verbose=False,
            pattern_merge_percent=1,
            char_len=len(text)
        )

        # Update vocab_size based on actual BPE vocabulary
        self.vocab_size = len(self.bpe.vocab)

        # Encode full text and split into train/val
        data = torch.tensor(self.bpe.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s: str) -> List[int]:
        """
        Encode a string using BPE

        Args:
            s: Input string to encode.
        """
        return self.bpe.encode(s)

    def save_encoder(self, filepath: str):
        """Save both encoder and decoder"""
        self.bpe.save_encoder(filepath + '.encoder')
        self.bpe.save_decoder(filepath + '.decoder')

    def load_encoder(self, filepath: str):
        """Load both encoder and decoder"""
        self.bpe = BPE()
        self.bpe.load_encoder(filepath + '.encoder')
        self.bpe.load_decoder(filepath + '.decoder')

    def decode(self, l: List[int]) -> str:
        """
        Decode a list of integers using BPE

        Args:
            l: Input list of integers to decode.
        """
        return self.bpe.decode(l)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data for training or validation.

        Args:
            split: 'train' or 'val' to indicate training or validation data.
        """
        data = self.train_data if split == 'train' else self.val_data
        if len(data) <= self.config.block_size:
            raise ValueError(f"Data length {len(data)} must be greater than block_size {self.config.block_size}")

        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
