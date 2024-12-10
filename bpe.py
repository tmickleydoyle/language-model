"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from collections import Counter
from typing import List, Tuple
import os

class BPE:
    """
    The Byte Pair Encoding (BPE) class.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_pairs(self, tokens: List[int]) -> Counter:
        """
        Get the pairs of characters in the tokens.

        Args:
            tokens (List[int]): The list of token ids.

        Returns:
            Counter: The pairs of characters.
        """
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[tokens[i], tokens[i + 1]] += 1
        return pairs

    def train(self, text: str, max_vocab_size: int = 100, verbose: bool = False, pattern_merge_percent: int = 2, char_len: int = 10000) -> None:
        """
        Train the BPE model on the given text.

        Args:
            text (str): The text to train the model on.
            max_vocab_size (int): The maximum vocabulary size.
            verbose (bool): Whether to print verbose output.
            pattern_merge_percent (int): The percentage threshold for including a merge. Defaul is 95% for the top 5% of pairs.
            char_len (int): The total number of characters from the text to consider.
        """
        assert max_vocab_size >= 0
        num_merges = max_vocab_size
        include_merge = int((pattern_merge_percent / 100) * char_len)
        assert include_merge >= 0

        # Input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255

        # Iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i in range(num_merges):
            # Count up the number of times every consecutive pair appears
            stats = self.get_pairs(ids)
            stats = {pair: count for pair, count in stats.items() if count >= include_merge}
            if not stats:
                break  # no more pairs to merge

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            # Mint a new token: assign it the next available id
            idx = 256 + i
            # Replace all occurrences of pair in ids with idx
            ids = self.merge(ids, pair, idx)
            # Save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # Prints
            if verbose:
                print(f"merge: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab    # used in decode()

    def merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge all occurrences of the given pair in the ids list.

        Args:
            ids (List[int]): The list of token ids.
            pair (Tuple[int, int]): The pair of token ids to merge.
            new_id (int): The new token id to replace the pair with.

        Returns:
            List[int]: The updated list of token ids.
        """
        i = 0
        while i < len(ids) - 1:
            if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                ids = ids[:i] + [new_id] + ids[i + 2:]
            else:
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode the given list of token ids into a string.

        Args:
            ids (List[int]): The list of token ids.

        Returns:
            str: The decoded string.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> List[int]:
        """
        Encode the given string into a list of token ids.

        Args:
            text (str): The string to encode.

        Returns:
            List[int]: The list of token ids.
        """
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            stats = self.get_pairs(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # Subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # We can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # Otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids
    
    def save_encoder(self, path: str = "encoder.txt") -> None:
        """
        Save the encoder (vocab and merges) to a file.

        Args:
            path (str): The path to save the encoder to.
        """
        with open(path, "w") as file:
            # Save vocabulary
            file.write("### VOCAB ###\n")
            for idx, token in self.vocab.items():
                file.write(f"{idx} {token.hex()}\n")
            # Save merges
            file.write("### MERGES ###\n")
            for (p1, p2), idx in self.merges.items():
                file.write(f"{p1} {p2} {idx}\n")

    def load_encoder(self, path: str = "encoder.txt") -> None:
        """
        Load the encoder from a file.

        Args:
            path (str): The path to load the encoder from.
        """
        self.vocab = {}
        self.merges = {}
        with open(path, "r") as file:
            section = ""
            for line in file:
                if line.startswith("###"):
                    section = line.strip()
                    continue
                if not line.strip():
                    continue
                if section == "### VOCAB ###":
                    idx, token_hex = line.strip().split()
                    self.vocab[int(idx)] = bytes.fromhex(token_hex)
                elif section == "### MERGES ###":
                    p1, p2, idx = line.strip().split()
                    self.merges[(int(p1), int(p2))] = int(idx)

    def save_decoder(self, path: str = "decoder.txt") -> None:
        """
        Save the decoder to a file.

        Args:
            path (str): The path to save the decoder to.
        """
        with open(path, "w") as file:
            for idx, token in self.vocab.items():
                file.write(f"{idx} {token.hex()}\n")

    def load_decoder(self, path: str = "decoder.txt") -> None:
        """
        Load the decoder from a file.

        Args:
            path (str): The path to load the decoder from.
        """
        self.vocab = {}
        with open(path, "r") as file:
            for line in file:
                idx, token_hex = line.strip().split()
                self.vocab[int(idx)] = bytes.fromhex(token_hex)

# # Get all files from the docs directory and concatenate their contents
# docs_dir = "example/"
# example_text = ""
# for filename in os.listdir(docs_dir):
#     if filename.endswith(".txt"):
#         with open(os.path.join(docs_dir, filename), "r") as file:
#             example_text += file.read() + "\n"

# # get number of unique terms in the text
# num_terms = len(set(example_text.split()))

# bpe = BPE()
# bpe.train(text=example_text, max_vocab_size=100, verbose=True, pattern_merge_percent=1, char_len=num_terms)
# encoded = bpe.encode(example_text)
# decoded = bpe.decode(encoded)

# # print(f"Encoded: {encoded}")
# # print(f"Decoded: {decoded}")
# # Print the vocabulary from most common to least common
# # print("Vocabulary:" + "\n".join(f"{k}: {v}" for k, v in sorted(bpe.vocab.items(), key=lambda x: x[0], reverse=False)))

# # print('Saving encoder and decoder to files')
# # bpe.save_encoder()
# # bpe.save_decoder()

# # print('Loading encoder and decoder from files')
# # new_bpe = BPE()
# # new_bpe.load_encoder()
# # new_bpe.load_decoder()
# # new_encoded = new_bpe.encode("Hello, world!")
# # new_decoded = new_bpe.decode(new_encoded)
# # print(f"New Encoded: {new_encoded}")
# # print(f"New Decoded: {new_decoded}")

