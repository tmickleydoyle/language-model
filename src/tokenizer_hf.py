"""HuggingFace tokenizer wrapper with same interface as BPETokenizer."""

from pathlib import Path
from tokenizers import Tokenizer


class HFTokenizer:
    """Wrapper around HuggingFace tokenizer with BPETokenizer-compatible interface."""

    # Special token IDs (matching the trainer config)
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1
    BOS_TOKEN_ID = 2
    EOS_TOKEN_ID = 3

    def __init__(self, tokenizer_path: str = None):
        """Initialize tokenizer, optionally loading from path."""
        self._tokenizer = None
        if tokenizer_path:
            self.load(tokenizer_path)

    def load(self, path: str):
        """Load tokenizer from HuggingFace JSON format."""
        path = Path(path)

        # Handle both directory and direct file path
        if path.is_dir():
            tokenizer_file = path / "tokenizer.json"
        elif path.suffix == ".json":
            tokenizer_file = path
        else:
            # Try adding .json
            tokenizer_file = Path(str(path) + ".json")
            if not tokenizer_file.exists():
                tokenizer_file = path / "tokenizer.json"

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")

        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Verify special tokens
        vocab = self._tokenizer.get_vocab()
        assert self.PAD_TOKEN in vocab, f"Missing {self.PAD_TOKEN} token"
        assert self.BOS_TOKEN in vocab, f"Missing {self.BOS_TOKEN} token"
        assert self.EOS_TOKEN in vocab, f"Missing {self.EOS_TOKEN} token"

        # Update token IDs from actual vocab
        self.PAD_TOKEN_ID = vocab[self.PAD_TOKEN]
        self.UNK_TOKEN_ID = vocab.get(self.UNK_TOKEN, 1)
        self.BOS_TOKEN_ID = vocab[self.BOS_TOKEN]
        self.EOS_TOKEN_ID = vocab[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.get_vocab_size()

    @property
    def vocab(self) -> dict:
        """Return vocabulary dict."""
        return self._tokenizer.get_vocab()

    @property
    def bos_token_id(self) -> int:
        """Return BOS token ID."""
        return self.BOS_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self.EOS_TOKEN_ID

    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        encoded = self._tokenizer.encode(text)
        return encoded.ids

    def decode(self, token_ids: list, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {self.PAD_TOKEN_ID, self.BOS_TOKEN_ID, self.EOS_TOKEN_ID}
            token_ids = [t for t in token_ids if t not in special_ids]
        return self._tokenizer.decode(token_ids)

    def encode_with_special_tokens(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list:
        """Encode text with optional BOS/EOS tokens."""
        tokens = self.encode(text)
        if add_bos:
            tokens = [self.BOS_TOKEN_ID] + tokens
        if add_eos:
            tokens = tokens + [self.EOS_TOKEN_ID]
        return tokens
