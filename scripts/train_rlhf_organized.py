#!/usr/bin/env python3
"""
Organized RLHF Training Script

This script trains GPT models using RLHF with the new organized data structure.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.model import create_model_factory as create_model
from src.model.reward_model import RewardModel
from src.tokenizer import BPETokenizer as DefaultTokenizer
from src.data.preference_dataset import PreferenceDataset, PreferenceCollector, create_synthetic_preferences
from src.rl.ppo import PPOTrainer, PPOConfig
from src.rl.dpo import DPOTrainer

logger = logging.getLogger(__name__)

# Define data paths
RLHF_DATA_DIR = project_root / "data" / "rlhf"
DATASETS_DIR = RLHF_DATA_DIR / "datasets"
PROMPTS_DIR = RLHF_DATA_DIR / "prompts"

# Predefined datasets
AVAILABLE_DATASETS = {
    "starter": DATASETS_DIR / "curated" / "preference_dataset_starter.json",
    "combined": DATASETS_DIR / "combined" / "combined_preference_dataset.json",
    "synthetic": DATASETS_DIR / "synthetic" / "large_synthetic_preferences.json",
    "instruction": DATASETS_DIR / "curated" / "preferences_instruction_following.json",
    "safety": DATASETS_DIR / "curated" / "preferences_safety.json",
    "reasoning": DATASETS_DIR / "curated" / "preferences_reasoning.json",
}

def load_dataset(dataset_name: str, tokenizer: DefaultTokenizer, max_length: int = 512) -> PreferenceDataset:
    """Load a dataset by name or path."""
    
    if dataset_name in AVAILABLE_DATASETS:
        dataset_path = AVAILABLE_DATASETS[dataset_name]
        print(f"📊 Loading predefined dataset: {dataset_name}")
    else:
        dataset_path = Path(dataset_name)
        print(f"📊 Loading custom dataset: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = PreferenceDataset.from_json(str(dataset_path), tokenizer, max_length=max_length)
    print(f"✅ Loaded {len(dataset)} examples from {dataset_path.name}")
    
    return dataset

def save_user_preferences(examples, output_dir: Path) -> Path:
    """Save user-collected preferences to organized structure."""
    
    user_dir = DATASETS_DIR / "user_collected"
    user_dir.mkdir(exist_ok=True)
    
    # Find next available filename
    counter = 1
    while True:
        filename = f"user_preferences_{counter:03d}.json"
        filepath = user_dir / filename
        if not filepath.exists():
            break
        counter += 1
    
    # Save dataset
    data = [example.to_dict() for example in examples]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved {len(examples)} user preferences to {filepath}")
    return filepath

def train_reward_model(
    config: Config,
    dataset_name: str,
    tokenizer: DefaultTokenizer,
    pretrained_path: Optional[str] = None,
    epochs: int = 3
) -> RewardModel:
    """Train a reward model on preference data."""
    print("🎯 Training Reward Model")
    print("=" * 50)
    
    # Load dataset with model's block_size as max_length
    dataset = load_dataset(dataset_name, tokenizer, max_length=config.block_size)
    train_dataset, val_dataset = dataset.split(0.8)
    
    # Load pretrained model if provided
    if pretrained_path:
        pretrained_model = create_model(config, config.vocab_size)
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
    save_path = Path("models") / "reward_model" / "model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": config.to_dict()
    }, save_path)
    print(f"✅ Saved reward model to {save_path}")
    
    return reward_model

def train_dpo(
    config: Config,
    model_path: str,
    dataset_name: str,
    tokenizer: DefaultTokenizer,
    epochs: int = 3
) -> None:
    """Train model using DPO."""
    print("⚡ Training with DPO")
    print("=" * 50)
    
    # Load dataset with model's block_size as max_length
    dataset = load_dataset(dataset_name, tokenizer, max_length=config.block_size)
    train_dataset, val_dataset = dataset.split(0.8)
    
    # Load model and reference model
    model = create_model(config, config.vocab_size)
    ref_model = create_model(config, config.vocab_size)
    
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
        save_path = Path("models") / "dpo_model" / f"epoch_{epoch+1}.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(save_path))
        
    # Save final model
    final_path = Path("models") / "dpo_model" / "final_model.pth"
    trainer.save_checkpoint(str(final_path))
    print(f"✅ Saved final DPO model to {final_path}")

def main():
    parser = argparse.ArgumentParser(description="Train GPT with organized RLHF data")
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
        "--dataset",
        type=str,
        default="starter",
        help="Dataset to use (starter|combined|synthetic|instruction|safety|reasoning) or custom path"
    )
    parser.add_argument(
        "--collect_preferences",
        action="store_true",
        help="Interactively collect preferences"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="collection_prompts.txt",
        help="Prompts file for collection (in prompts/ directory)"
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
    
    # Load config from checkpoint first
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if "config" in checkpoint:
        # Load saved config
        config = Config.from_dict(checkpoint["config"])
        # Override with command line args
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        if args.device:
            config.device = args.device
    else:
        # Fallback to default config with overrides
        config = Config(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device
        )
    
    # Load tokenizer
    model_dir = Path(args.model_path).parent
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = DefaultTokenizer()
    
    # Load tokenizer from JSON file
    import json
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Reconstruct vocab and merges
    tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in tokenizer_data['vocab'].items()}
    tokenizer.merges = {}
    for merge_key, idx in tokenizer_data['merges'].items():
        p1, p2 = merge_key.split(',')
        tokenizer.merges[(int(p1), int(p2))] = idx
    
    # Build encoder/decoder from vocab
    tokenizer.encoder = {}
    tokenizer.decoder = {}
    for token_id, token_bytes in tokenizer.vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        tokenizer.encoder[token_str] = token_id
        tokenizer.decoder[token_id] = token_str
    
    # Handle preference collection
    if args.collect_preferences:
        print("🤝 Interactive Preference Collection")
        print("=" * 50)
        
        # Load model for generation
        model = create_model(config, config.vocab_size)
        checkpoint = torch.load(args.model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.device)
        
        collector = PreferenceCollector(model, tokenizer, config.device)
        
        # Load prompts
        prompts_path = PROMPTS_DIR / args.prompts
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            print(f"⚠️  Prompts file not found: {prompts_path}")
            prompts = [
                "Explain how machine learning works: ",
                "Write a short story about robots: ",
                "What are the benefits of exercise? ",
            ]
        
        # Collect preferences
        examples = []
        for prompt in prompts[:5]:  # Limit to first 5 for demo
            example = collector.collect_preference(prompt)
            if example:
                examples.append(example)
        
        # Save collected preferences
        if examples:
            dataset_path = save_user_preferences(examples, DATASETS_DIR)
            args.dataset = str(dataset_path)  # Use collected data for training
        else:
            print("❌ No preferences collected, using default dataset")
    
    # Show available datasets
    print("📊 Available predefined datasets:")
    for name, path in AVAILABLE_DATASETS.items():
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"   {name}: {len(data)} examples")
    
    # Train with specified method
    if args.method == "dpo":
        train_dpo(config, args.model_path, args.dataset, tokenizer, args.epochs)
    else:
        print("⚠️  PPO training not implemented in this organized version yet")
        print("Use DPO instead: --method dpo")
        
    print("🎉 RLHF training complete!")

if __name__ == "__main__":
    main()