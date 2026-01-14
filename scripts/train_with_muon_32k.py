#!/usr/bin/env python3
"""Train GPT model using Muon optimizer.

This script trains a GPT language model using the Muon optimizer, which
applies Newton-Schulz orthogonalization to gradient updates for 2D hidden
layer parameters. Uses existing tokenizer from prose_model_fresh.

Usage:
    python scripts/train_with_muon.py
"""

import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
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
        logging.FileHandler('muon_training.log')
    ]
)
logger = logging.getLogger(__name__)


class FastTokenizedDataset:
    """Simple dataset that works with pre-tokenized data."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def get_batch(self, batch_size: int, device: str):
        """Get a random batch of data."""
        ix = torch.randint(len(self.tokens) - self.block_size, (batch_size,))
        x = torch.stack([self.tokens[i:i+self.block_size] for i in ix])
        y = torch.stack([self.tokens[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)


def label_smoothed_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Compute cross entropy loss with label smoothing.

    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar loss value
    """
    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)

    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = F.nll_loss(log_probs, targets, reduction='mean')

    # Smooth loss: uniform distribution over all classes
    smooth_loss = -log_probs.mean(dim=-1).mean()

    # Combine: (1 - smoothing) * nll + smoothing * uniform
    loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
    return loss


def get_lr_with_warmup_cosine(step: int, warmup_steps: int, total_steps: int,
                               max_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Get learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        max_lr: Maximum learning rate (after warmup)
        min_lr_ratio: Ratio of min LR to max LR (default: 0.1 = decay to 10%)

    Returns:
        Learning rate for current step
    """
    min_lr = max_lr * min_lr_ratio

    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def encode_story_files(data_dir: Path, tokenizer: HFTokenizer, file_pattern: str = "story_*.txt") -> list:
    """Encode individual story files, each as a complete document with BOS/EOS.

    Args:
        data_dir: Directory containing story files
        tokenizer: BPE tokenizer
        file_pattern: Glob pattern for story files

    Returns:
        List of token IDs with BOS/EOS markers around each story
    """
    story_files = sorted(data_dir.glob(file_pattern))
    total_files = len(story_files)

    if total_files == 0:
        raise ValueError(f"No files matching '{file_pattern}' found in {data_dir}")

    all_tokens = []
    logger.info(f"Encoding {total_files:,} story files...")
    start_time = time.time()

    bos_id = tokenizer.BOS_TOKEN_ID
    eos_id = tokenizer.EOS_TOKEN_ID

    for i, story_file in enumerate(story_files):
        # Read complete story
        with open(story_file, 'r', encoding='utf-8') as f:
            story_text = f.read().strip()

        if not story_text:
            continue

        # Encode this story
        story_tokens = tokenizer.encode(story_text)

        # Wrap with BOS/EOS for complete story
        all_tokens.append(bos_id)
        all_tokens.extend(story_tokens)
        all_tokens.append(eos_id)

        # Progress every 5000 files
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            files_per_sec = (i + 1) / elapsed
            remaining = (total_files - i - 1) / files_per_sec
            logger.info(f"  Encoded {i+1:,}/{total_files:,} files ({files_per_sec:.0f} files/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    logger.info(f"Encoding complete: {len(all_tokens):,} tokens from {total_files:,} files in {elapsed:.1f}s")

    return all_tokens


def fast_encode_documents(text: str, tokenizer: HFTokenizer, doc_separator: str = "\n\n") -> list:
    """Encode text by processing each document separately (much faster for large texts).

    Args:
        text: Full text with documents separated by doc_separator
        tokenizer: BPE tokenizer
        doc_separator: String that separates documents

    Returns:
        List of token IDs with BOS/EOS markers
    """
    documents = text.split(doc_separator)
    documents = [doc.strip() for doc in documents if doc.strip()]

    all_tokens = []
    total_docs = len(documents)

    logger.info(f"Encoding {total_docs:,} documents...")
    start_time = time.time()

    bos_id = tokenizer.BOS_TOKEN_ID
    eos_id = tokenizer.EOS_TOKEN_ID

    for i, doc in enumerate(documents):
        if not doc:
            continue

        # Encode this document
        doc_tokens = tokenizer.encode(doc)

        # Add BOS at start, tokens, EOS at end
        all_tokens.append(bos_id)
        all_tokens.extend(doc_tokens)
        all_tokens.append(eos_id)

        # Progress every 1000 docs
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            docs_per_sec = (i + 1) / elapsed
            remaining = (total_docs - i - 1) / docs_per_sec
            logger.info(f"  Encoded {i+1:,}/{total_docs:,} docs ({docs_per_sec:.0f} docs/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    logger.info(f"Encoding complete: {len(all_tokens):,} tokens in {elapsed:.1f}s")

    return all_tokens


def main():
    """Main training function with Muon optimizer."""
    print("=" * 60)
    print("GPT Training with Muon Optimizer")
    print("=" * 60)

    # Configuration
    DATA_DIR = PROJECT_ROOT / "data" / "tinystories_500k"
    TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer_32k"
    OUTPUT_DIR = PROJECT_ROOT / "muon_model_32k"

    # Training hyperparameters
    EPOCHS = 250
    BATCH_SIZE = 8  # Reduced for MPS memory
    CONTEXT_SIZE = 512
    EMBEDDING_DIM = 384
    NUM_HEADS = 8
    NUM_LAYERS = 6

    # Muon-specific hyperparameters (optimized for 250 epochs)
    MUON_LR = 0.002  # Lower LR for stability with large dataset
    ADAMW_LR = 2e-4  # Lower LR for embeddings/output
    MOMENTUM = 0.95
    NS_STEPS = 5

    # Optimization improvements (Round 4 - 250 epochs)
    WEIGHT_DECAY = 0.1   # Stronger regularization for large dataset
    LABEL_SMOOTHING = 0.0  # No smoothing for lowest possible loss
    GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 128 (8 * 16)
    EARLY_STOP_PATIENCE = 20  # More patience for slow convergence
    WARMUP_RATIO = 0.02  # ~5 epochs warmup
    DROPOUT = 0.2  # More dropout to prevent overfitting

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

    # Check for pre-tokenized data first (much faster loading)
    TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized_500k_32k"
    train_pt = TOKENIZED_DIR / "train_tokens.pt"
    val_pt = TOKENIZED_DIR / "val_tokens.pt"

    if train_pt.exists() and val_pt.exists():
        # Fast path: load pre-tokenized data
        logger.info(f"Loading pre-tokenized data from {TOKENIZED_DIR}...")
        train_tokens = torch.load(train_pt)
        val_tokens = torch.load(val_pt)
        logger.info(f"Loaded training tokens: {len(train_tokens):,}")
        logger.info(f"Loaded validation tokens: {len(val_tokens):,}")
    else:
        # Slow path: encode from raw files
        logger.info("Pre-tokenized data not found. Encoding from raw files...")
        logger.info("  (Run scripts/pretokenize_data.py for faster future runs)")

        story_files = sorted(DATA_DIR.glob("story_*.txt"))
        total_stories = len(story_files)
        logger.info(f"Found {total_stories:,} story files in {DATA_DIR}")

        if total_stories == 0:
            raise ValueError(f"No story files found in {DATA_DIR}")

        # Split files into train/val (90/10)
        split_idx = int(total_stories * 0.9)
        train_files = story_files[:split_idx]
        val_files = story_files[split_idx:]
        logger.info(f"Train: {len(train_files):,} stories, Val: {len(val_files):,} stories")

        # Encode training stories (each file = complete story with BOS/EOS)
        logger.info("Encoding training stories...")
        bos_id = tokenizer.BOS_TOKEN_ID
        eos_id = tokenizer.EOS_TOKEN_ID

        train_tokens = []
        for i, story_file in enumerate(train_files):
            with open(story_file, 'r', encoding='utf-8') as f:
                story_text = f.read().strip()
            if story_text:
                train_tokens.append(bos_id)
                train_tokens.extend(tokenizer.encode(story_text))
                train_tokens.append(eos_id)
            if (i + 1) % 5000 == 0:
                logger.info(f"  Encoded {i+1:,}/{len(train_files):,} training stories")

        train_tokens = torch.tensor(train_tokens, dtype=torch.long)
        logger.info(f"Training tokens: {len(train_tokens):,}")

        # Encode validation stories
        logger.info("Encoding validation stories...")
        val_tokens = []
        for story_file in val_files:
            with open(story_file, 'r', encoding='utf-8') as f:
                story_text = f.read().strip()
            if story_text:
                val_tokens.append(bos_id)
                val_tokens.extend(tokenizer.encode(story_text))
                val_tokens.append(eos_id)

        val_tokens = torch.tensor(val_tokens, dtype=torch.long)
        logger.info(f"Validation tokens: {len(val_tokens):,}")

    # Create datasets
    train_dataset = FastTokenizedDataset(train_tokens, block_size=CONTEXT_SIZE)
    val_dataset = FastTokenizedDataset(val_tokens, block_size=CONTEXT_SIZE)

    logger.info(f"Train dataset: {len(train_dataset):,} samples")
    logger.info(f"Val dataset: {len(val_dataset):,} samples")

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
    logger.info(f"Model: {format_parameter_count(total_params)} params ({format_parameter_count(trainable_params)} trainable)")

    # Create Muon optimizer
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
    logger.info("Starting training with Muon optimizer (Round 2)")
    logger.info(f"  Weight decay: {WEIGHT_DECAY}, Label smoothing: {LABEL_SMOOTHING}")
    logger.info(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps (effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    logger.info(f"  LR warmup: {WARMUP_RATIO*100:.0f}% of steps, Early stopping patience: {EARLY_STOP_PATIENCE}")
    logger.info("=" * 60)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_start_time = time.time()

    # Calculate total steps for LR scheduling
    num_batches_per_epoch = min(500, len(train_dataset) // BATCH_SIZE)
    total_steps = EPOCHS * num_batches_per_epoch // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    global_step = 0

    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()

        # Training phase with gradient accumulation
        total_train_loss = 0.0
        num_batches = min(500, len(train_dataset) // BATCH_SIZE)
        optimizer.zero_grad()

        for batch_idx in range(num_batches):
            # Get batch
            x, y = train_dataset.get_batch(batch_size=BATCH_SIZE, device=device)

            # Forward pass - get logits only (we'll compute loss with label smoothing)
            logits, _ = model(x, y)

            # Compute loss with label smoothing
            loss = label_smoothed_cross_entropy(logits, y, smoothing=LABEL_SMOOTHING)

            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()

            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # Optimizer step every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update learning rates with warmup + cosine decay
                muon_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, MUON_LR)
                adamw_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, ADAMW_LR)
                optimizer.lr = muon_lr
                optimizer.adamw_lr = adamw_lr

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Progress logging
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                current_lr = get_lr_with_warmup_cosine(global_step, warmup_steps, total_steps, MUON_LR)
                logger.info(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{num_batches} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        avg_train_loss = total_train_loss / num_batches

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        num_val_batches = min(100, len(val_dataset) // BATCH_SIZE)

        with torch.no_grad():
            for _ in range(num_val_batches):
                x, y = val_dataset.get_batch(batch_size=BATCH_SIZE, device=device)
                _, loss = model(x, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        epoch_time = time.time() - epoch_start_time

        # Compute perplexity
        train_ppl = min(10000, 2.71828 ** avg_train_loss)
        val_ppl = min(10000, 2.71828 ** avg_val_loss)

        logger.info(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} (ppl: {train_ppl:.1f}) | "
            f"Val Loss: {avg_val_loss:.4f} (ppl: {val_ppl:.1f}) | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model and check for early stopping
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
                logger.info(f"  Early stopping triggered after {epoch+1} epochs")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'config': config.to_dict(),
                'vocab_size': vocab_size,
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")

            # Generate sample text
            model.eval()
            try:
                prompt = "Once upon a time"
                encoded = tokenizer.encode(prompt)
                input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

                with torch.no_grad():
                    generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
                    generated_text = tokenizer.decode(generated[0].tolist())
                    display_text = generated_text[:200].replace('\n', ' ')
                    logger.info(f"  Sample: {display_text}...")
            except Exception as e:
                logger.warning(f"  Sample generation failed: {e}")

    # Training complete
    total_time = time.time() - train_start_time
    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Copy tokenizer to output directory
    import shutil
    shutil.copy(str(TOKENIZER_PATH) + ".vocab", OUTPUT_DIR / "tokenizer.vocab")
    shutil.copy(str(TOKENIZER_PATH) + ".merges", OUTPUT_DIR / "tokenizer.merges")
    logger.info(f"Tokenizer copied to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
