"""Direct Preference Optimization (DPO) implementation.

DPO is a simpler alternative to PPO that directly optimizes for human preferences
without needing a separate reward model or RL loop.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.gpt import ModernGPTLanguageModel as GPT
from ..config import Config

logger = logging.getLogger(__name__)


class DPOTrainer:
    """Direct Preference Optimization trainer.
    
    DPO simplifies RLHF by directly optimizing the policy to match
    human preferences without needing a reward model or PPO.
    
    Args:
        model: Language model to fine-tune
        ref_model: Reference model (frozen copy of initial model)
        config: Model configuration
        beta: KL penalty coefficient (default: 0.1)
    """
    
    def __init__(
        self,
        model: GPT,
        ref_model: GPT,
        config: Config,
        beta: float = 0.1
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.beta = beta
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def compute_log_probs(
        self,
        model: GPT,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities for sequences.
        
        Args:
            model: Model to use for computation
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Attention mask
            
        Returns:
            Log probabilities for each token
        """
        outputs = model(input_ids)
        # Handle both tuple and dict outputs
        if isinstance(outputs, tuple):
            logits = outputs[0]  # First element is logits
        else:
            logits = outputs["logits"]
        
        # Shift for autoregressive modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
            token_log_probs = token_log_probs * shift_mask
            
        return token_log_probs
    
    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss.
        
        Args:
            chosen_ids: Token indices for preferred responses
            rejected_ids: Token indices for rejected responses
            chosen_mask: Attention mask for chosen sequences
            rejected_mask: Attention mask for rejected sequences
            
        Returns:
            Loss value and metrics dictionary
        """
        # Compute log probabilities for policy model
        chosen_log_probs = self.compute_log_probs(
            self.model, chosen_ids, chosen_mask
        )
        rejected_log_probs = self.compute_log_probs(
            self.model, rejected_ids, rejected_mask
        )
        
        # Compute log probabilities for reference model
        with torch.no_grad():
            ref_chosen_log_probs = self.compute_log_probs(
                self.ref_model, chosen_ids, chosen_mask
            )
            ref_rejected_log_probs = self.compute_log_probs(
                self.ref_model, rejected_ids, rejected_mask
            )
        
        # Sum log probs over sequence length
        if chosen_mask is not None:
            chosen_log_probs = (chosen_log_probs * chosen_mask[:, 1:]).sum(dim=1)
            ref_chosen_log_probs = (ref_chosen_log_probs * chosen_mask[:, 1:]).sum(dim=1)
        else:
            chosen_log_probs = chosen_log_probs.sum(dim=1)
            ref_chosen_log_probs = ref_chosen_log_probs.sum(dim=1)
            
        if rejected_mask is not None:
            rejected_log_probs = (rejected_log_probs * rejected_mask[:, 1:]).sum(dim=1)
            ref_rejected_log_probs = (ref_rejected_log_probs * rejected_mask[:, 1:]).sum(dim=1)
        else:
            rejected_log_probs = rejected_log_probs.sum(dim=1)
            ref_rejected_log_probs = ref_rejected_log_probs.sum(dim=1)
        
        # Compute log ratios
        chosen_log_ratio = chosen_log_probs - ref_chosen_log_probs
        rejected_log_ratio = rejected_log_probs - ref_rejected_log_probs
        
        # DPO loss
        losses = -F.logsigmoid(self.beta * (chosen_log_ratio - rejected_log_ratio))
        loss = losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = ((chosen_log_ratio - rejected_log_ratio) > 0).float().mean()
            chosen_rewards = self.beta * chosen_log_ratio
            rejected_rewards = self.beta * rejected_log_ratio
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def train_step(
        self,
        chosen_batch: Dict[str, torch.Tensor],
        rejected_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute one DPO training step.
        
        Args:
            chosen_batch: Batch of preferred responses
            rejected_batch: Batch of rejected responses
            
        Returns:
            Dictionary of training metrics
        """
        # Extract inputs
        chosen_ids = chosen_batch["input_ids"]
        rejected_ids = rejected_batch["input_ids"]
        chosen_mask = chosen_batch.get("attention_mask", None)
        rejected_mask = rejected_batch.get("attention_mask", None)
        
        # Compute loss
        loss, metrics = self.compute_dpo_loss(
            chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        
        self.optimizer.step()
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "beta": self.beta
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved DPO checkpoint to {path}")
        
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.beta = checkpoint.get("beta", 0.1)
        logger.info(f"Loaded DPO checkpoint from {path}")