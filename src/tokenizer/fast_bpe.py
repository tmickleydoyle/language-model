"""Fast BPE tokenizer implementation with optimizations."""

import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import bisect


class FastBPETokenizer:
    """Optimized BPE tokenizer with O(n log n) encoding."""
    
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        # Cache for fast merge lookups
        self._merge_priority: Dict[Tuple[int, int], int] = {}
        
    def train(self, text: str, max_vocab_size: int = 10000):
        """Train BPE with optimized pair counting."""
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Use a heap for efficient best pair selection
        pair_heap = []
        pair_counts = defaultdict(int)
        pair_positions = defaultdict(set)
        
        # Initial pair counting with position tracking
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1
            pair_positions[pair].add(i)
        
        # Initialize heap (negative count for max heap)
        for pair, count in pair_counts.items():
            heapq.heappush(pair_heap, (-count, pair))
        
        # Perform merges
        num_merges = max_vocab_size - 256
        for merge_idx in range(num_merges):
            # Find best pair
            while pair_heap:
                neg_count, best_pair = heapq.heappop(pair_heap)
                if best_pair in pair_counts and pair_counts[best_pair] == -neg_count:
                    break
            else:
                break  # No more pairs to merge
                
            if pair_counts[best_pair] < 2:
                break
                
            # Create new token
            new_token_id = 256 + merge_idx
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Update tokens and counts efficiently
            positions = sorted(pair_positions[best_pair])
            offset = 0
            
            for pos in positions:
                actual_pos = pos - offset
                if actual_pos < len(tokens) - 1 and tokens[actual_pos] == best_pair[0] and tokens[actual_pos + 1] == best_pair[1]:
                    # Replace pair with new token
                    tokens[actual_pos] = new_token_id
                    del tokens[actual_pos + 1]
                    offset += 1
                    
                    # Update affected pairs
                    if actual_pos > 0:
                        old_pair = (tokens[actual_pos - 1], best_pair[0])
                        new_pair = (tokens[actual_pos - 1], new_token_id)
                        self._update_pair_count(pair_counts, pair_positions, old_pair, -1, actual_pos - 1)
                        self._update_pair_count(pair_counts, pair_positions, new_pair, 1, actual_pos - 1)
                        heapq.heappush(pair_heap, (-pair_counts[new_pair], new_pair))
                    
                    if actual_pos < len(tokens) - 1:
                        old_pair = (best_pair[1], tokens[actual_pos + 1])
                        new_pair = (new_token_id, tokens[actual_pos + 1])
                        self._update_pair_count(pair_counts, pair_positions, old_pair, -1, actual_pos)
                        self._update_pair_count(pair_counts, pair_positions, new_pair, 1, actual_pos)
                        heapq.heappush(pair_heap, (-pair_counts[new_pair], new_pair))
            
            # Remove the merged pair
            del pair_counts[best_pair]
            del pair_positions[best_pair]
        
        # Build merge priority cache
        self._merge_priority = {pair: idx for idx, pair in enumerate(self.merges.keys())}
    
    def _update_pair_count(self, counts, positions, pair, delta, pos):
        """Update pair count and position tracking."""
        if delta > 0:
            counts[pair] += delta
            positions[pair].add(pos)
        else:
            counts[pair] += delta
            if counts[pair] <= 0:
                del counts[pair]
                del positions[pair]
            else:
                positions[pair].discard(pos)
    
    def encode(self, text: str) -> List[int]:
        """Encode text with O(n log n) complexity using priority queue."""
        tokens = list(text.encode('utf-8'))
        
        if not self.merges:
            return tokens
            
        # Use a min heap for merge priorities
        merge_heap = []
        token_positions = list(range(len(tokens)))  # Track original positions
        
        # Find all possible merges
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self._merge_priority:
                # Priority, position, pair
                heapq.heappush(merge_heap, (self._merge_priority[pair], i, pair))
        
        # Apply merges in order
        merged = set()  # Track positions that have been merged
        
        while merge_heap:
            priority, pos, pair = heapq.heappop(merge_heap)
            
            # Skip if position already merged
            if pos in merged or pos + 1 in merged:
                continue
                
            # Verify the pair still exists at this position
            if pos >= len(tokens) - 1 or tokens[pos] != pair[0] or tokens[pos + 1] != pair[1]:
                continue
            
            # Apply merge
            new_token = self.merges[pair]
            tokens[pos] = new_token
            del tokens[pos + 1]
            
            # Mark positions as merged
            merged.add(pos)
            merged.add(pos + 1)
            
            # Adjust positions after deletion
            for i in range(len(token_positions)):
                if token_positions[i] > pos:
                    token_positions[i] -= 1
            
            # Add new possible merges
            if pos > 0 and pos - 1 not in merged:
                new_pair = (tokens[pos - 1], new_token)
                if new_pair in self._merge_priority:
                    heapq.heappush(merge_heap, (self._merge_priority[new_pair], pos - 1, new_pair))
            
            if pos < len(tokens) - 1:
                new_pair = (new_token, tokens[pos + 1])
                if new_pair in self._merge_priority:
                    heapq.heappush(merge_heap, (self._merge_priority[new_pair], pos, new_pair))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        text_bytes = b''.join(self.vocab[token_id] for token_id in token_ids)
        return text_bytes.decode('utf-8', errors='replace')