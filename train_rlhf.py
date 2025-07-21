#!/usr/bin/env python3
"""
Train GPT model using Reinforcement Learning from Human Feedback (RLHF).

This script supports both PPO and DPO algorithms for fine-tuning language models
based on human preferences.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import create_model
from src.model.reward_model import RewardModel
from src.tokenizer import DefaultTokenizer
from src.data.preference_dataset import PreferenceDataset, PreferenceCollector, create_synthetic_preferences
from src.rl.ppo import PPOTrainer, PPOConfig
from src.rl.dpo import DPOTrainer

logger = logging.getLogger(__name__)


def train_reward_model(
    config: Config,
    train_dataset: PreferenceDataset,
    val_dataset: Optional[PreferenceDataset] = None,
    pretrained_path: Optional[str] = None,
    epochs: int = 3
) -> RewardModel:
    """Train a reward model on preference data."""
    print("🎯 Training Reward Model")
    print("=" * 50)
    
    # Load pretrained model if provided
    if pretrained_path:
        pretrained_model = create_model(config)
        checkpoint = torch.load(pretrained_path, map_location=config.device)
        pretrained_model.load_state_dict(checkpoint["model_state_dict"])
        reward_model = RewardModel(config, pretrained_model)
    else:
        reward_model = RewardModel(config)
        
    reward_model = reward_model.to(config.device)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            chosen_ids = batch["chosen_ids"].to(config.device)
            rejected_ids = batch["rejected_ids"].to(config.device)
            chosen_mask = batch["chosen_mask"].to(config.device)
            rejected_mask = batch["rejected_mask"].to(config.device)
            
            # Compute loss
            loss, metrics = reward_model.compute_preference_loss(
                chosen_ids, rejected_ids, chosen_mask, rejected_mask
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reward_model.parameters(),
                config.grad_clip
            )
            optimizer.step()
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Accuracy={metrics['accuracy']:.4f}")
                
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, "
              f"Avg Accuracy: {avg_accuracy:.4f}")
        
    # Save reward model
    save_path = Path("reward_model") / "model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": config.to_dict()
    }, save_path)
    print(f"✅ Saved reward model to {save_path}")
    
    return reward_model


def train_ppo(
    config: Config,
    policy_model_path: str,
    reward_model: RewardModel,
    prompts: list,
    tokenizer: DefaultTokenizer,
    steps: int = 1000
) -> None:
    """Train model using PPO."""
    print("🚀 Training with PPO")
    print("=" * 50)
    
    # Load policy and reference models
    policy_model = create_model(config)
    ref_model = create_model(config)
    
    # Load pretrained weights
    checkpoint = torch.load(policy_model_path, map_location=config.device)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    
    policy_model = policy_model.to(config.device)
    ref_model = ref_model.to(config.device)
    
    # Create PPO trainer
    ppo_config = PPOConfig()
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        config=config,
        ppo_config=ppo_config
    )
    
    # Training loop
    for step in range(steps):
        # Sample batch of prompts
        batch_size = min(config.batch_size, len(prompts))
        batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]
        
        # Train step
        metrics = trainer.train_step(
            batch_prompts,
            tokenizer,
            max_length=256
        )
        
        if step % 10 == 0:
            print(f"Step {step}: "
                  f"Policy Loss={metrics['policy_loss']:.4f}, "
                  f"KL={metrics['kl_div']:.4f}, "
                  f"Entropy={metrics['entropy']:.4f}")
            
        if step % 100 == 0:
            # Save checkpoint
            save_path = Path("ppo_model") / f"checkpoint_{step}.pth"
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                "model_state_dict": policy_model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": config.to_dict(),
                "step": step
            }, save_path)
            
    # Save final model
    final_path = Path("ppo_model") / "final_model.pth"
    torch.save({
        "model_state_dict": policy_model.state_dict(),
        "config": config.to_dict()
    }, final_path)
    print(f"✅ Saved final PPO model to {final_path}")


def train_dpo(
    config: Config,
    model_path: str,
    train_dataset: PreferenceDataset,
    val_dataset: Optional[PreferenceDataset] = None,
    epochs: int = 3
) -> None:
    """Train model using DPO."""
    print("⚡ Training with DPO")
    print("=" * 50)
    
    # Load model and reference model
    model = create_model(config)
    ref_model = create_model(config)
    
    # Load pretrained weights
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ref_model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(config.device)
    ref_model = ref_model.to(config.device)
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        beta=0.1
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            chosen_batch = {
                "input_ids": batch["chosen_ids"].to(config.device),
                "attention_mask": batch["chosen_mask"].to(config.device)
            }
            rejected_batch = {
                "input_ids": batch["rejected_ids"].to(config.device),
                "attention_mask": batch["rejected_mask"].to(config.device)
            }
            
            # Train step
            metrics = trainer.train_step(chosen_batch, rejected_batch)
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Accuracy={metrics['accuracy']:.4f}, "
                      f"Reward Margin={metrics['reward_margin']:.4f}")
                      
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, "
              f"Avg Accuracy: {avg_accuracy:.4f}")
              
        # Save checkpoint
        save_path = Path("dpo_model") / f"epoch_{epoch+1}.pth"
        save_path.parent.mkdir(exist_ok=True)
        trainer.save_checkpoint(str(save_path))
        
    # Save final model
    final_path = Path("dpo_model") / "final_model.pth"
    trainer.save_checkpoint(str(final_path))
    print(f"✅ Saved final DPO model to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GPT with RLHF")
    parser.add_argument(
        "--method",
        type=str,
        choices=["ppo", "dpo"],
        default="dpo",
        help="RL method to use (PPO or DPO)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--preference_data",
        type=str,
        help="Path to preference dataset JSON file"
    )
    parser.add_argument(
        "--collect_preferences",
        action="store_true",
        help="Interactively collect preferences"
    )
    parser.add_argument(
        "--synthetic_preferences",
        action="store_true",
        help="Generate synthetic preferences"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = Config(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Load tokenizer
    model_dir = Path(args.model_path).parent
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = DefaultTokenizer.load(str(tokenizer_path))
    
    # Handle preference data
    if args.collect_preferences:
        # Interactive collection
        model = create_model(config)
        checkpoint = torch.load(args.model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)
        
        collector = PreferenceCollector(model, tokenizer, config.device)
        
        prompts = []
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default prompts
            prompts = [
                "Write a short story about a robot learning to paint: ",
                "Explain quantum computing to a 5-year-old: ",
                "What are the benefits of renewable energy? ",
                "Describe the perfect pizza: ",
                "How do you train for a marathon? "
            ]
            
        dataset = collector.collect_batch(prompts, "preferences.json")
        preference_data_path = "preferences.json"
    elif args.synthetic_preferences:
        # Generate synthetic preferences
        model = create_model(config)
        checkpoint = torch.load(args.model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)
        
        prompts = []
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = ["Tell me a joke: ", "What is AI? ", "How to bake a cake? "]
            
        examples = create_synthetic_preferences(
            model, tokenizer, prompts, device=config.device
        )
        dataset = PreferenceDataset(examples, tokenizer)
        dataset.save_json("synthetic_preferences.json")
        preference_data_path = "synthetic_preferences.json"
    else:
        preference_data_path = args.preference_data
        
    # Load preference dataset
    dataset = PreferenceDataset.from_json(preference_data_path, tokenizer)
    train_dataset, val_dataset = dataset.split(0.8)
    
    print(f"📊 Loaded {len(train_dataset)} training examples")
    print(f"📊 Loaded {len(val_dataset)} validation examples")
    
    if args.method == "ppo":
        # Train reward model first
        reward_model = train_reward_model(
            config, train_dataset, val_dataset, args.model_path, args.epochs
        )
        
        # Load prompts for PPO
        prompts = []
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Extract prompts from preference data
            prompts = list(set([ex.prompt for ex in dataset.examples]))
            
        # Train with PPO
        train_ppo(
            config, args.model_path, reward_model, prompts, tokenizer, steps=1000
        )
    else:
        # Train with DPO
        train_dpo(
            config, args.model_path, train_dataset, val_dataset, args.epochs
        )
        
    print("🎉 RLHF training complete!")


if __name__ == "__main__":
    main()