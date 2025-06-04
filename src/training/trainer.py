"""
Training and evaluation utilities for the GPT language model.

This module provides classes and functions for training the model,
evaluating performance, and managing the training loop.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
from torch.amp import GradScaler, autocast

from ..model import GPTLanguageModel, create_model
from ..data import TextDataset
from ..utils import count_parameters, format_parameter_count
from ..config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for GPT language model.

    Handles the training loop, evaluation, model saving/loading,
    and progress tracking.

    Args:
        model: GPT model to train
        config: Configuration object
        train_dataset: Training dataset (optional)
        val_dataset: Validation dataset (optional)
        save_dir: Directory to save checkpoints
        optimizer: PyTorch optimizer (optional, defaults to AdamW)
    """

    def __init__(self,
                 model: GPTLanguageModel,
                 config: Config,
                 train_dataset: Optional[TextDataset] = None,
                 val_dataset: Optional[TextDataset] = None,
                 save_dir: Optional[Union[str, Path]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Initialize the trainer."""
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.device

        # Set save directory
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = Path(".")

        # Initialize optimizer
        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Initialize gradient scaler for mixed precision
        self.scaler: GradScaler = GradScaler('cuda' if config.fp16 else 'cpu', enabled=config.fp16)

        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create dataloaders if datasets are provided
        self.train_dataloader = train_dataset if train_dataset else None
        self.val_dataloader = val_dataset if val_dataset else None

        total_params, trainable_params = count_parameters(model)
        logger.info(
            f"Trainer initialized for model with "
            f"{format_parameter_count(total_params)} parameters "
            f"({format_parameter_count(trainable_params)} trainable)"
        )

    def train_step(self) -> float:
        """
        Perform a single training step.

        Returns:
            Training loss for this step
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        self.model.train()

        # Get batch from training dataset
        x, y = self.train_dataset.get_batch(
            batch_size=self.config.batch_size,
            device=self.config.get_device()
        )

        # Forward pass with mixed precision
        device = 'cuda' if self.config.fp16 else 'cpu'
        with autocast(device, enabled=self.config.fp16):
            logits, loss = self.model(x, y)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return float(loss.item())

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform a single training step with given input and target tensors.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Training loss for this step
        """
        self.model.train()

        # Forward pass with mixed precision
        device = 'cuda' if self.config.fp16 else 'cpu'
        with autocast(device, enabled=self.config.fp16):
            logits, loss = self.model(x, y)

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.global_step += 1
        return float(loss.item())

    @torch.no_grad()
    def evaluate(self) -> Optional[Union[Dict[str, float], float]]:
        """
        Evaluate the model on train and validation sets.

        Returns:
            Dictionary with train and validation losses, or single float for val_loss
        """
        self.model.eval()

        eval_results = {}

        # Evaluate on training set
        if self.train_dataset is not None:
            losses = []
            for _ in range(self.config.eval_iters):
                try:
                    x, y = self.train_dataset.get_batch(
                        batch_size=self.config.batch_size,
                        device=self.config.get_device()
                    )

                    device = 'cuda' if self.config.fp16 else 'cpu'
                    with autocast(device, enabled=self.config.fp16):
                        logits, loss = self.model(x, y)

                    losses.append(loss.item())

                except Exception as e:
                    logger.warning(f"Error during train evaluation: {e}")
                    continue

            if losses:
                eval_results['train_loss'] = sum(losses) / len(losses)
            else:
                eval_results['train_loss'] = float('inf')

        # Evaluate on validation set
        if self.val_dataset is not None:
            losses = []
            for _ in range(self.config.eval_iters):
                try:
                    x, y = self.val_dataset.get_batch(
                        batch_size=self.config.batch_size,
                        device=self.config.get_device()
                    )

                    device = 'cuda' if self.config.fp16 else 'cpu'
                    with autocast(device, enabled=self.config.fp16):
                        logits, loss = self.model(x, y)

                    losses.append(loss.item())

                except Exception as e:
                    logger.warning(f"Error during val evaluation: {e}")
                    continue

            if losses:
                eval_results['val_loss'] = sum(losses) / len(losses)
            else:
                eval_results['val_loss'] = float('inf')

        # Return single float for backward compatibility if only val_dataset
        if len(eval_results) == 1 and 'val_loss' in eval_results:
            val_loss: float = eval_results['val_loss']
            return val_loss

        # Return None if no datasets available
        if not eval_results:
            return None

        return eval_results

    def train(self) -> Optional[Dict[str, float]]:
        """
        Run the complete training loop.

        Returns:
            Final evaluation results
        """
        logger.info("Starting training...")
        start_time = time.time()

        # Determine training mode: iterations or epochs
        max_epochs = getattr(self.config, 'max_epochs', None)
        max_iters = getattr(self.config, 'max_iters', 10000)

        try:
            if max_epochs is not None:
                # Epoch-based training
                for epoch in range(max_epochs):
                    self.current_epoch = epoch
                    avg_loss = self.train_epoch()

                    # Evaluation
                    is_eval_epoch = (
                        epoch % self.config.eval_interval == 0
                        or epoch == max_epochs - 1
                    )
                    if is_eval_epoch:
                        eval_results = self.evaluate()

                        # Log progress
                        elapsed = time.time() - start_time
                        if isinstance(eval_results, dict):
                            train_loss = eval_results.get('train_loss', avg_loss)
                            val_loss_log = eval_results.get('val_loss', 0)
                        else:
                            train_loss = avg_loss
                            val_loss_log = eval_results if eval_results is not None else 0

                        logger.info(
                            f"Epoch {epoch:3d} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss_log:.4f} | "
                            f"Time: {elapsed:.1f}s"
                        )

                        # Save best model
                        if isinstance(eval_results, dict):
                            val_loss = eval_results.get('val_loss', float('inf'))
                        else:
                            val_loss = eval_results if eval_results is not None else float('inf')

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(self.save_dir / 'best_model.pth')
                            logger.info(
                                f"New best model saved with val loss: {val_loss:.4f}")

                    # Save regular checkpoints based on save_interval
                    save_interval = getattr(
                        self.config, 'save_interval', self.config.eval_interval)
                    if epoch % save_interval == 0 or epoch == max_epochs - 1:
                        self.save_checkpoint()  # This will use default naming
            else:
                # Iteration-based training (legacy)
                for iteration in range(max_iters):
                    self.current_iter = iteration

                    # Training step
                    self.train_step()

                    # Evaluation
                    is_eval_iter = (
                        iteration % self.config.eval_interval == 0
                        or iteration == max_iters - 1
                    )
                    if is_eval_iter:
                        eval_results = self.evaluate()

                        # Log progress
                        elapsed = time.time() - start_time
                        if isinstance(eval_results, dict):
                            train_loss_log = eval_results.get('train_loss', 0)
                            val_loss_log = eval_results.get('val_loss', 0)
                        else:
                            train_loss_log = 0
                            val_loss_log = eval_results if eval_results is not None else 0

                        logger.info(
                            f"Step {iteration:5d} | "
                            f"Train Loss: {train_loss_log:.4f} | "
                            f"Val Loss: {val_loss_log:.4f} | "
                            f"Time: {elapsed:.1f}s"
                        )

                        # Save best model
                        if isinstance(eval_results, dict):
                            val_loss = eval_results.get('val_loss', float('inf'))
                        else:
                            val_loss = eval_results if eval_results is not None else float('inf')

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(self.save_dir / 'best_model.pth')
                            logger.info(
                                f"New best model saved with val loss: {val_loss:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Final evaluation
        final_results = self.evaluate()

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final results: {final_results}")

        if isinstance(final_results, dict):
            return final_results
        elif final_results is not None:
            return {'val_loss': final_results}
        else:
            return None

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        if self.train_dataset is None:
            raise RuntimeError("No training dataset provided")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Simple epoch training (could be enhanced with proper DataLoader)
        for _ in range(min(self.config.max_batches_per_epoch, len(self.train_dataset))):  # Limit batches per epoch
            try:
                loss = self.train_step()
                total_loss += loss
                num_batches += 1
                self.global_step += 1
            except Exception as e:
                logger.warning(f"Error in training step: {e}")
                continue

        avg_loss = total_loss / max(1, num_batches)
        self.current_epoch += 1

        return avg_loss

    def save_checkpoint(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save model checkpoint.

        Args:
            file_path: Path to save the checkpoint (optional)

        Returns:
            Path where checkpoint was saved
        """
        if file_path is None:
            file_path = self.save_dir / f"checkpoint_step_{self.global_step}.pth"
        else:
            file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.to_dict(),
                'current_iter': self.current_iter,
                'current_epoch': self.current_epoch,
                'epoch': self.current_epoch,  # Add epoch key for backward compatibility
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
            }

            # Add vocab size if available
            if self.train_dataset is not None:
                checkpoint['vocab_size'] = self.train_dataset.tokenizer.vocab_size
            elif self.val_dataset is not None:
                checkpoint['vocab_size'] = self.val_dataset.tokenizer.vocab_size

            torch.save(checkpoint, file_path)
            logger.info(f"Saved checkpoint to {file_path}")

            return file_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, file_path: Union[str, Path]) -> None:
        """
        Load model checkpoint.

        Args:
            file_path: Path to the checkpoint file
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")

        try:
            checkpoint = torch.load(file_path, map_location=self.config.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_iter = checkpoint.get('current_iter', 0)
            self.current_epoch = checkpoint.get('current_epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            logger.info(f"Checkpoint loaded from {file_path}")
            logger.info(
                f"Resumed from iteration {self.current_iter}, "
                f"epoch {self.current_epoch}"
            )

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def generate_text(self,
                      prompt: str,
                      tokenizer: Any,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_k: Optional[int] = None) -> str:
        """
        Generate text from a prompt using the trained model.

        Args:
            prompt: Text prompt to start generation
            tokenizer: Tokenizer to use for encoding/decoding
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated text including the prompt

        Raises:
            ValueError: If tokenizer is None
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for text generation")

        # Use config defaults if not specified
        if max_tokens is None:
            max_tokens = self.config.default_max_tokens
        if temperature is None:
            temperature = self.config.default_temperature
        if top_k is None:
            top_k = self.config.default_top_k

        self.model.eval()

        try:
            # Encode prompt
            encoded_prompt = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor(
                [encoded_prompt],
                dtype=torch.long,
                device=self.device
            )

            # Generate
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    generated_ids = self.model.generate(
                        prompt_tensor,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                else:
                    # Simple fallback generation
                    generated_ids = prompt_tensor.clone()
                    vocab_size = self.config.vocab_size

                    for _ in range(max_tokens):
                        device = 'cuda' if self.config.fp16 else 'cpu'
                        with autocast(device, enabled=self.config.fp16):
                            logits, _ = self.model(generated_ids)

                        # Get last token logits and sample
                        logits = logits[:, -1, :] / temperature

                        # Ensure we don't sample tokens outside vocab
                        logits = logits[:, :vocab_size]

                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('inf')

                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Decode with error handling
            try:
                generated_text = tokenizer.decode(generated_ids[0].tolist())
                return str(generated_text)
            except Exception:
                # Fallback: try to decode up to the problematic token
                token_ids = generated_ids[0].tolist()
                decoded_parts = []
                for i, token_id in enumerate(token_ids):
                    try:
                        decoded_parts.append(tokenizer.decode([token_id]))
                    except Exception:
                        # Skip problematic tokens
                        logger.warning(f"Skipping token ID {token_id} during decoding")
                        continue
                return str(''.join(decoded_parts))

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation failed: {e}"

    def generate_sample(self,
                        prompt: str,
                        max_new_tokens: Optional[int] = None,
                        temperature: Optional[float] = None,
                        top_k: Optional[int] = None) -> str:
        """
        Generate text sample from a prompt (legacy method - use generate_text instead).

        Args:
            prompt: Text prompt to start generation
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated text including the prompt
        """
        # Use config defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.config.default_max_tokens
        if temperature is None:
            temperature = self.config.default_temperature
        if top_k is None:
            top_k = self.config.default_top_k

        # Get tokenizer from dataset
        tokenizer = None
        if self.train_dataset is not None:
            tokenizer = self.train_dataset.tokenizer
        elif self.val_dataset is not None:
            tokenizer = self.val_dataset.tokenizer
        else:
            raise ValueError(
                "No tokenizer available - provide dataset or use "
                "generate_text with tokenizer"
            )

        return self.generate_text(prompt, tokenizer, max_new_tokens, temperature, top_k)


def create_trainer(config: Config, data_file: str) -> Trainer:
    """Create a trainer with model and dataset.

    Args:
        config: Configuration object
        data_file: Path to training data file

    Returns:
        Configured trainer instance
    """
    # Load dataset
    dataset = TextDataset()
    dataset.load_data(data_file)

    # Create model
    model = create_model(config, dataset.vocab_size)

    # Create trainer
    trainer = Trainer(model, config, train_dataset=dataset)

    return trainer
