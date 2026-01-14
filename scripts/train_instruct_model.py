#!/usr/bin/env python3
"""Train instruction-following model with masked loss.

Only computes loss on output tokens (### Response: section).
"""

import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.model import create_model
from src.tokenizer_hf import HFTokenizer
from src.training import create_muon_optimizer
from src.utils import count_parameters, format_parameter_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('instruct_training.log')
    ]
)
logger = logging.getLogger(__name__)


class InstructDataset:
    """Dataset for instruction data with loss masks."""

    def __init__(self, data_path: Path, context_size: int):
        data = torch.load(data_path)
        self.tokens_list = data['tokens']
        self.masks_list = data['masks']
        self.context_size = context_size
        self.pad_token_id = 0  # <pad>

    def __len__(self):
        return len(self.tokens_list)

    def get_batch(self, batch_size: int, device: str):
        """Get a random batch of padded sequences with loss masks."""
        indices = torch.randint(len(self.tokens_list), (batch_size,))

        # Get sequences and masks
        batch_tokens = []
        batch_masks = []

        for idx in indices:
            tokens = self.tokens_list[idx]
            mask = self.masks_list[idx]

            # Truncate if too long
            if len(tokens) > self.context_size + 1:
                tokens = tokens[:self.context_size + 1]
                mask = mask[:self.context_size + 1]

            # Pad if too short
            if len(tokens) < self.context_size + 1:
                pad_len = self.context_size + 1 - len(tokens)
                tokens = tokens + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len  # Don't compute loss on padding

            batch_tokens.append(tokens)
            batch_masks.append(mask)

        # Convert to tensors
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        masks_tensor = torch.tensor(batch_masks, dtype=torch.float)

        # Input is all but last token, target is all but first token
        x = tokens_tensor[:, :-1].to(device)
        y = tokens_tensor[:, 1:].to(device)
        loss_mask = masks_tensor[:, 1:].to(device)  # Align with targets

        return x, y, loss_mask


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss only on masked tokens.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        mask: (batch, seq_len) - 1 for tokens to include in loss, 0 otherwise
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for cross-entropy
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    # Compute per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Apply mask
    masked_loss = loss_per_token * mask_flat

    # Average over masked tokens
    num_masked = mask_flat.sum()
    if num_masked > 0:
        loss = masked_loss.sum() / num_masked
    else:
        loss = masked_loss.sum()  # Fallback

    return loss


def get_lr_with_warmup_cosine(step: int, warmup_steps: int, total_steps: int,
                               max_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Get learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    min_lr = max_lr * min_lr_ratio
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    print("=" * 60)
    print("Instruction-Following Model Training")
    print("=" * 60)

    # Configuration
    TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer_instruct"
    DATA_DIR = PROJECT_ROOT / "data" / "tokenized_instruct_1m"
    OUTPUT_DIR = PROJECT_ROOT / "instruct_model"

    # Training hyperparameters
    EPOCHS = 250
    BATCH_SIZE = 4  # Smaller batch for longer context
    CONTEXT_SIZE = 1024  # Longer context for instructions
    EMBEDDING_DIM = 384
    NUM_HEADS = 8
    NUM_LAYERS = 12  # Deeper model for better capacity

    # Muon-specific hyperparameters
    MUON_LR = 0.001  # Lower LR for stability
    ADAMW_LR = 1e-4
    MOMENTUM = 0.95
    NS_STEPS = 5

    # Optimization
    WEIGHT_DECAY = 0.1
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 32
    EARLY_STOP_PATIENCE = 25  # More patience for longer training
    WARMUP_RATIO = 0.02
    DROPOUT = 0.1  # Lower dropout for larger model

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Metal Performance Shaders")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = HFTokenizer(str(TOKENIZER_PATH))
    vocab_size = tokenizer.vocab_size
    logger.info(f"Tokenizer loaded: {vocab_size} tokens")

    # Load data
    logger.info(f"Loading data from {DATA_DIR}...")
    train_dataset = InstructDataset(DATA_DIR / "train.pt", CONTEXT_SIZE)
    val_dataset = InstructDataset(DATA_DIR / "val.pt", CONTEXT_SIZE)
    logger.info(f"Train: {len(train_dataset):,} examples")
    logger.info(f"Val: {len(val_dataset):,} examples")

    # Create config
    config = Config(
        vocab_size=vocab_size,
        n_embd=EMBEDDING_DIM,
        n_head=NUM_HEADS,
        n_layer=NUM_LAYERS,
        block_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCHS,
        device=device,
        dropout=DROPOUT,
        learning_rate=ADAMW_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config, vocab_size)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model: {format_parameter_count(total_params)} params")

    # Create optimizer
    logger.info("Creating Muon optimizer...")
    optimizer = create_muon_optimizer(
        model,
        muon_lr=MUON_LR,
        adamw_lr=ADAMW_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        ns_steps=NS_STEPS,
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training with masked loss")
    logger.info(f"  Context: {CONTEXT_SIZE}, Batch: {BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info("=" * 60)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_start_time = time.time()

    # Calculate total steps
    num_batches_per_epoch = min(1000, len(train_dataset) // BATCH_SIZE)
    total_steps = EPOCHS * num_batches_per_epoch // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    global_step = 0

    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()

        total_train_loss = 0.0
        num_batches = min(1000, len(train_dataset) // BATCH_SIZE)
        optimizer.zero_grad()

        for batch_idx in range(num_batches):
            # Get batch with loss mask
            x, y, loss_mask = train_dataset.get_batch(batch_size=BATCH_SIZE, device=device)

            # Forward pass
            logits, _ = model(x, y)

            # Compute masked loss
            loss = masked_cross_entropy(logits, y, loss_mask)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward
            loss.backward()
            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Optimizer step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                muon_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, MUON_LR)
                adamw_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, ADAMW_LR)
                optimizer.lr = muon_lr
                optimizer.adamw_lr = adamw_lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                current_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, MUON_LR)
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{num_batches} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        avg_train_loss = total_train_loss / num_batches

        # Validation
        model.eval()
        total_val_loss = 0.0
        num_val_batches = min(200, len(val_dataset) // BATCH_SIZE)

        with torch.no_grad():
            for _ in range(num_val_batches):
                x, y, loss_mask = val_dataset.get_batch(batch_size=BATCH_SIZE, device=device)
                logits, _ = model(x, y)
                loss = masked_cross_entropy(logits, y, loss_mask)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        epoch_time = time.time() - epoch_start_time

        train_ppl = min(10000, 2.71828 ** avg_train_loss)
        val_ppl = min(10000, 2.71828 ** avg_val_loss)

        logger.info(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} (ppl: {train_ppl:.1f}) | "
            f"Val Loss: {avg_val_loss:.4f} (ppl: {val_ppl:.1f}) | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            checkpoint_path = OUTPUT_DIR / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'config': config.to_dict(),
                'vocab_size': vocab_size,
            }, checkpoint_path)
            logger.info(f"  New best model saved (val_loss: {avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                logger.info(f"  Early stopping triggered")
                break

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'config': config.to_dict(),
                'vocab_size': vocab_size,
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved")

    # Done
    total_time = time.time() - train_start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time/60:.1f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

    # Copy tokenizer
    import shutil
    tokenizer_src = TOKENIZER_PATH / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, OUTPUT_DIR / "tokenizer.json")
        logger.info("Tokenizer copied to output directory")


if __name__ == "__main__":
    main()
