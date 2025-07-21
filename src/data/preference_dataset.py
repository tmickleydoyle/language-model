"""Preference dataset for RLHF training.

This module provides utilities for creating, loading, and managing
preference datasets used in reinforcement learning from human feedback.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PreferenceExample:
    """Single preference comparison example."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceExample":
        """Create from dictionary."""
        return cls(**data)


class PreferenceDataset(Dataset):
    """Dataset for preference comparisons.
    
    Stores examples where each example contains a prompt and two responses,
    one chosen (preferred) and one rejected.
    
    Args:
        examples: List of preference examples
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        examples: List[PreferenceExample],
        tokenizer: Any,
        max_length: int = 512
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]
        
        # Combine prompt with responses
        chosen_text = example.prompt + example.chosen
        rejected_text = example.prompt + example.rejected
        
        # Tokenize based on tokenizer type
        if hasattr(self.tokenizer, 'encode'):
            # Custom BPE tokenizer
            chosen_ids = self.tokenizer.encode(chosen_text)
            rejected_ids = self.tokenizer.encode(rejected_text)
            
            # Truncate if needed
            if len(chosen_ids) > self.max_length:
                chosen_ids = chosen_ids[:self.max_length]
            if len(rejected_ids) > self.max_length:
                rejected_ids = rejected_ids[:self.max_length]
                
            # Pad to max_length
            chosen_mask = [1] * len(chosen_ids)
            rejected_mask = [1] * len(rejected_ids)
            
            # Pad with zeros (assuming 0 is pad token)
            pad_token = 0
            while len(chosen_ids) < self.max_length:
                chosen_ids.append(pad_token)
                chosen_mask.append(0)
            while len(rejected_ids) < self.max_length:
                rejected_ids.append(pad_token)
                rejected_mask.append(0)
                
            # Convert to tensors
            chosen_encoding = {
                "input_ids": torch.tensor(chosen_ids).unsqueeze(0),
                "attention_mask": torch.tensor(chosen_mask).unsqueeze(0)
            }
            rejected_encoding = {
                "input_ids": torch.tensor(rejected_ids).unsqueeze(0),
                "attention_mask": torch.tensor(rejected_mask).unsqueeze(0)
            }
        else:
            # HuggingFace tokenizer
            chosen_encoding = self.tokenizer(
                chosen_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            rejected_encoding = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        return {
            "chosen_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_mask": rejected_encoding["attention_mask"].squeeze(0),
            "prompt_length": len(self.tokenizer.encode(example.prompt)) if hasattr(self.tokenizer, 'encode') else len(self.tokenizer(example.prompt)["input_ids"])
        }
    
    @classmethod
    def from_json(cls, file_path: str, tokenizer: Any, max_length: int = 512) -> "PreferenceDataset":
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        examples = [PreferenceExample.from_dict(item) for item in data]
        return cls(examples, tokenizer, max_length)
    
    def save_json(self, file_path: str) -> None:
        """Save dataset to JSON file."""
        data = [example.to_dict() for example in self.examples]
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def split(self, train_ratio: float = 0.8) -> Tuple["PreferenceDataset", "PreferenceDataset"]:
        """Split into train and validation sets."""
        split_idx = int(len(self.examples) * train_ratio)
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]
        
        train_dataset = PreferenceDataset(train_examples, self.tokenizer, self.max_length)
        val_dataset = PreferenceDataset(val_examples, self.tokenizer, self.max_length)
        
        return train_dataset, val_dataset


class PreferenceCollector:
    """Interactive preference collection tool.
    
    Helps collect human preferences by generating multiple responses
    and asking for comparisons.
    """
    
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.examples: List[PreferenceExample] = []
        
    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 2,
        max_length: int = 128,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> List[str]:
        """Generate multiple responses for a prompt."""
        responses = []
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]
        
        for _ in range(num_responses):
            with torch.no_grad():
                # Generate with different random seeds for diversity
                torch.manual_seed(torch.randint(0, 10000, (1,)).item())
                
                output = self.model.generate(
                    input_ids,
                    max_length=prompt_length + max_length,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode and extract only the generated part
            full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = full_text[len(prompt):]
            responses.append(response.strip())
            
        return responses
    
    def collect_preference(
        self,
        prompt: str,
        responses: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[PreferenceExample]:
        """Collect a single preference comparison."""
        if responses is None:
            responses = self.generate_responses(prompt)
            
        if len(responses) < 2:
            logger.error("Need at least 2 responses for comparison")
            return None
            
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        for i, response in enumerate(responses):
            print(f"\nResponse {i + 1}:")
            print(response)
            print("-" * 50)
            
        # Get user preference
        while True:
            try:
                choice = input(f"\nWhich response is better? (1-{len(responses)}): ")
                chosen_idx = int(choice) - 1
                if 0 <= chosen_idx < len(responses):
                    break
                print(f"Please enter a number between 1 and {len(responses)}")
            except ValueError:
                print("Please enter a valid number")
                
        # Select a rejected response (not the chosen one)
        rejected_idx = (chosen_idx + 1) % len(responses)
        
        example = PreferenceExample(
            prompt=prompt,
            chosen=responses[chosen_idx],
            rejected=responses[rejected_idx],
            metadata=metadata
        )
        
        self.examples.append(example)
        return example
    
    def collect_batch(
        self,
        prompts: List[str],
        save_path: Optional[str] = None
    ) -> PreferenceDataset:
        """Collect preferences for multiple prompts."""
        for prompt in prompts:
            self.collect_preference(prompt)
            
        dataset = PreferenceDataset(self.examples, self.tokenizer)
        
        if save_path:
            dataset.save_json(save_path)
            logger.info(f"Saved {len(self.examples)} preferences to {save_path}")
            
        return dataset
    
    def load_examples(self, file_path: str) -> None:
        """Load previously collected examples."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        self.examples = [PreferenceExample.from_dict(item) for item in data]
        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")


def create_synthetic_preferences(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    reward_model: Optional[Any] = None,
    num_responses_per_prompt: int = 4,
    device: str = "cpu"
) -> List[PreferenceExample]:
    """Create synthetic preferences using a reward model.
    
    This is useful for bootstrapping when human preferences are limited.
    
    Args:
        model: Language model for generating responses
        tokenizer: Tokenizer
        prompts: List of prompts
        reward_model: Model to score responses (if None, uses length as proxy)
        num_responses_per_prompt: Number of responses to generate per prompt
        device: Device to run on
        
    Returns:
        List of preference examples
    """
    examples = []
    
    for prompt in prompts:
        # Generate multiple responses
        collector = PreferenceCollector(model, tokenizer, device)
        responses = collector.generate_responses(
            prompt,
            num_responses=num_responses_per_prompt
        )
        
        if reward_model is not None:
            # Score with reward model
            scores = []
            for response in responses:
                full_text = prompt + response
                input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = reward_model(input_ids)
                    score = outputs["rewards"].item()
                    
                scores.append(score)
        else:
            # Use simple heuristics (e.g., length, no repetition)
            scores = []
            for response in responses:
                # Prefer longer, non-repetitive responses
                score = len(response.split())
                # Penalize repetition
                unique_words = len(set(response.lower().split()))
                score *= (unique_words / len(response.split()))
                scores.append(score)
                
        # Select best and worst as chosen/rejected
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        
        if best_idx != worst_idx:
            example = PreferenceExample(
                prompt=prompt,
                chosen=responses[best_idx],
                rejected=responses[worst_idx],
                metadata={"scores": scores}
            )
            examples.append(example)
            
    return examples