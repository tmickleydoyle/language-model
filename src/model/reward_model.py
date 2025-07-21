"""Reward Model for Reinforcement Learning from Human Feedback (RLHF).

This module implements a reward model that predicts human preferences between
different model outputs, used for training language models with RL techniques.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import ModernGPTLanguageModel as GPT
from ..config import Config

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Reward model for predicting human preferences.
    
    Uses a pre-trained GPT model as backbone and adds a reward head
    to predict scalar rewards for generated sequences.
    
    Args:
        config: Model configuration
        pretrained_model: Optional pre-trained GPT model to use as backbone
    """
    
    def __init__(
        self, 
        config: Config,
        pretrained_model: Optional[GPT] = None
    ) -> None:
        super().__init__()
        self.config = config
        
        # Use pretrained model or create new one
        if pretrained_model is not None:
            self.backbone = pretrained_model
            # Freeze backbone initially
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone = GPT(config)
            
        # Reward head: projects from hidden states to scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, 1)
        )
        
        # Value head for advantage estimation (used in PPO)
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize reward and value head weights."""
        for module in [self.reward_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_values: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through reward model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_values: Whether to also return value estimates
            
        Returns:
            Dictionary containing:
                - rewards: Scalar rewards for each sequence
                - values: Value estimates (if return_values=True)
                - hidden_states: Last hidden states from backbone
        """
        # Get hidden states from backbone
        # For now, we'll need to get the embeddings manually since the model doesn't return hidden states
        # We'll use the model's internal representations
        x = self.backbone.token_embedding(input_ids)
        
        # Pass through transformer blocks
        for block in self.backbone.blocks:
            x, _ = block(x)
            
        # Apply final norm
        hidden_states = self.backbone.norm(x)
        
        # Use last non-padded token's hidden state for each sequence
        if attention_mask is not None:
            # Find last real token position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_idx, seq_lengths]
        else:
            # Use last token
            last_hidden = hidden_states[:, -1, :]
            
        # Compute rewards
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        results = {
            "rewards": rewards,
            "hidden_states": hidden_states
        }
        
        # Compute values if requested
        if return_values:
            values = self.value_head(last_hidden).squeeze(-1)
            results["values"] = values
            
        return results
    
    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
        margin: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute preference ranking loss.
        
        Args:
            chosen_ids: Token indices for preferred responses
            rejected_ids: Token indices for rejected responses
            chosen_mask: Attention mask for chosen sequences
            rejected_mask: Attention mask for rejected sequences
            margin: Minimum margin between chosen and rejected rewards
            
        Returns:
            Loss value and metrics dictionary
        """
        # Get rewards for both chosen and rejected
        chosen_outputs = self.forward(chosen_ids, chosen_mask)
        rejected_outputs = self.forward(rejected_ids, rejected_mask)
        
        chosen_rewards = chosen_outputs["rewards"]
        rejected_rewards = rejected_outputs["rewards"]
        
        # Ranking loss with margin
        losses = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin))
        loss = losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item()
        }
        
        return loss, metrics
    
    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the top.
                       If None, unfreezes all layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze top N transformer blocks
            for i, block in enumerate(self.backbone.transformer.h):
                if i >= len(self.backbone.transformer.h) - num_layers:
                    for param in block.parameters():
                        param.requires_grad = True