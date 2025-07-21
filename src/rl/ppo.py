"""Proximal Policy Optimization (PPO) implementation for RLHF.

This module implements PPO algorithm for fine-tuning language models
using reinforcement learning from human feedback.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..model.gpt import ModernGPTLanguageModel as GPT
from ..model.reward_model import RewardModel
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    kl_penalty_weight: float = 0.1
    entropy_weight: float = 0.01
    value_loss_weight: float = 0.5
    max_grad_norm: float = 0.5
    adaptive_kl_ctrl: bool = True
    target_kl: float = 0.01
    kl_penalty_init: float = 0.1


class PPOTrainer:
    """PPO trainer for language model fine-tuning.
    
    Implements the PPO algorithm with KL penalty for preventing
    the policy from deviating too far from the reference model.
    
    Args:
        policy_model: The language model being fine-tuned
        ref_model: Reference model for KL penalty (frozen copy)
        reward_model: Model that provides reward signals
        config: Model configuration
        ppo_config: PPO-specific configuration
    """
    
    def __init__(
        self,
        policy_model: GPT,
        ref_model: GPT,
        reward_model: RewardModel,
        config: Config,
        ppo_config: Optional[PPOConfig] = None
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        self.ppo_config = ppo_config or PPOConfig()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Adaptive KL controller
        if self.ppo_config.adaptive_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                init_kl_coef=self.ppo_config.kl_penalty_init,
                target=self.ppo_config.target_kl
            )
        else:
            self.kl_ctl = FixedKLController(self.ppo_config.kl_penalty_weight)
            
        # Optimizer for policy model
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute rewards for generated sequences using reward model."""
        with torch.no_grad():
            outputs = self.reward_model(input_ids, attention_mask)
            return outputs["rewards"]
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards for each timestep
            values: Value estimates for each timestep
            dones: Episode termination flags
            
        Returns:
            Advantages and returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.ppo_config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.ppo_config.gamma * self.ppo_config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
    
    def compute_policy_loss(
        self,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO policy loss with KL penalty.
        
        Args:
            input_ids: Input token indices
            actions: Selected actions (next tokens)
            old_log_probs: Log probabilities from old policy
            advantages: Computed advantages
            attention_mask: Attention mask
            
        Returns:
            Loss and metrics dictionary
        """
        # Get current policy outputs
        outputs = self.policy_model(input_ids)
        logits = outputs["logits"]
        
        # Compute log probabilities for taken actions
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_config.clip_epsilon, 1 + self.ppo_config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute KL divergence with reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids)
            ref_logits = ref_outputs["logits"]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
        kl_div = F.kl_div(log_probs, ref_log_probs.exp(), reduction='batchmean')
        
        # Add KL penalty
        kl_penalty = self.kl_ctl.value * kl_div
        
        # Entropy bonus for exploration
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()
        entropy_bonus = self.ppo_config.entropy_weight * entropy
        
        # Total loss
        total_loss = policy_loss + kl_penalty - entropy_bonus
        
        # Update KL controller
        self.kl_ctl.update(kl_div.item())
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "kl_penalty": kl_penalty.item(),
            "entropy": entropy.item(),
            "mean_ratio": ratio.mean().item(),
            "kl_coef": self.kl_ctl.value
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        prompts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """Execute one PPO training step.
        
        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer for encoding/decoding
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary of training metrics
        """
        # Generate responses from current policy
        responses = []
        old_log_probs_list = []
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.config.device)
            
            # Generate with policy model
            with torch.no_grad():
                output_ids, log_probs = self.generate_with_log_probs(
                    input_tensor,
                    max_length=max_length,
                    temperature=temperature
                )
                
            responses.append(output_ids)
            old_log_probs_list.append(log_probs)
            
        # Get rewards from reward model
        rewards_list = []
        values_list = []
        
        for response in responses:
            with torch.no_grad():
                outputs = self.reward_model(response, return_values=True)
                rewards_list.append(outputs["rewards"])
                values_list.append(outputs["values"])
                
        # Prepare data for PPO updates
        all_metrics = {}
        
        # Multiple PPO epochs
        for _ in range(self.ppo_config.ppo_epochs):
            epoch_metrics = []
            
            for i, (response, old_log_probs) in enumerate(zip(responses, old_log_probs_list)):
                # Compute advantages
                rewards = rewards_list[i]
                values = values_list[i]
                dones = torch.zeros_like(rewards)
                dones[-1] = 1  # Episode ends
                
                advantages, returns = self.compute_advantages(rewards, values, dones)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute loss
                loss, metrics = self.compute_policy_loss(
                    response[:-1],  # Input tokens
                    response[1:],   # Target tokens
                    old_log_probs,
                    advantages[:-1]
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.ppo_config.max_grad_norm
                )
                self.optimizer.step()
                
                epoch_metrics.append(metrics)
                
            # Average metrics for this epoch
            for key in epoch_metrics[0].keys():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(
                    sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
                )
                
        # Average across all PPO epochs
        final_metrics = {
            key: sum(values) / len(values)
            for key, values in all_metrics.items()
        }
        
        return final_metrics
    
    def generate_with_log_probs(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens and return log probabilities.
        
        Args:
            input_ids: Starting token indices
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token indices and their log probabilities
        """
        generated = input_ids
        log_probs_list = []
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.policy_model(generated)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                next_token = dist.sample()
                
                # Get log probability of sampled token
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_prob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
                log_probs_list.append(token_log_prob)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)
                
                # Stop if EOS token
                if hasattr(self.policy_model, 'eos_token_id') and next_token.item() == self.policy_model.eos_token_id:
                    break
                    
        log_probs = torch.cat(log_probs_list)
        return generated, log_probs


class AdaptiveKLController:
    """Adaptive KL penalty controller."""
    
    def __init__(self, init_kl_coef: float, target: float, horizon: int = 10000):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon
        
    def update(self, current: float, n_steps: int = 1):
        """Update KL coefficient based on current KL divergence."""
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        self.value *= 1 + proportional_error * n_steps / self.horizon


class FixedKLController:
    """Fixed KL penalty controller."""
    
    def __init__(self, kl_coef: float):
        self.value = kl_coef
        
    def update(self, current: float, n_steps: int = 1):
        """No-op for fixed controller."""
        pass