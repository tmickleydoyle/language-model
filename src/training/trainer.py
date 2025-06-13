"""Training and evaluation utilities for the GPT language model.

This module provides comprehensive training functionality including:
- Training loop management with epoch and iteration-based modes
- Mixed precision training support with automatic gradient scaling
- Model evaluation on train/validation sets
- Checkpoint saving and loading with state preservation
- Text generation capabilities during training
- Progress tracking and logging
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

import torch
from torch.amp import GradScaler, autocast

from ..model import GPTLanguageModel, create_model
from ..data import TextDataset
from ..utils import count_parameters, format_parameter_count
from ..config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for GPT language model.

    Handles the complete training pipeline including:
    - Training loop management with flexible epoch/iteration modes
    - Mixed precision training with automatic gradient scaling
    - Model evaluation and validation
    - Checkpoint management and state persistence
    - Text generation capabilities
    - Progress tracking and logging

    Args:
        model: GPT model to train
        config: Configuration object containing training parameters
        train_dataset: Training dataset (optional)
        val_dataset: Validation dataset (optional)
        save_dir: Directory to save checkpoints
        optimizer: PyTorch optimizer (optional, defaults to AdamW)
    """

    def __init__(
        self,
        model: GPTLanguageModel,
        config: Config,
        train_dataset: Optional[TextDataset] = None,
        val_dataset: Optional[TextDataset] = None,
        save_dir: Optional[Union[str, Path]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Initialize the trainer with model and configuration."""
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = config.device

        self._setup_save_directory(save_dir)
        self._initialize_optimizer(optimizer)
        self._initialize_training_state()
        self._log_initialization_info()

    def _setup_save_directory(self, save_dir: Optional[Union[str, Path]]) -> None:
        """Set up the directory for saving checkpoints."""
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = Path(".")

    def _initialize_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        """Initialize the optimizer with default AdamW if none provided."""
        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer

    def _initialize_training_state(self) -> None:
        """Initialize training state variables and gradient scaler."""
        # Initialize gradient scaler for mixed precision
        self.scaler: GradScaler = GradScaler(
            "cuda" if self.config.fp16 else "cpu", enabled=self.config.fp16
        )

        # Training state
        self.current_iter = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create dataloaders if datasets are provided
        self.train_dataloader = self.train_dataset if self.train_dataset else None
        self.val_dataloader = self.val_dataset if self.val_dataset else None

    def _log_initialization_info(self) -> None:
        """Log information about the initialized trainer."""
        total_params, trainable_params = count_parameters(self.model)
        logger.info(
            f"Trainer initialized for model with "
            f"{format_parameter_count(total_params)} parameters "
            f"({format_parameter_count(trainable_params)} trainable)"
        )

    def train_step(self) -> float:
        """Perform a single training step using the training dataset.

        Returns:
            Training loss for this step

        Raises:
            ValueError: If no training dataset is provided
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        self.model.train()

        # Get batch from training dataset
        x, y = self.train_dataset.get_batch(
            batch_size=self.config.batch_size, device=self.config.get_device()
        )

        return self._forward_backward_step(x, y)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Perform a single training step with given input and target tensors.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Training loss for this step
        """
        self.model.train()
        loss = self._forward_backward_step(x, y)
        self.global_step += 1
        return loss

    def _forward_backward_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Execute forward and backward pass with mixed precision support."""
        # Forward pass with mixed precision
        device = "cuda" if self.config.fp16 else "cpu"
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

    @torch.no_grad()
    def evaluate(self) -> Optional[Union[Dict[str, float], float]]:
        """Evaluate the model on train and validation sets.

        Returns:
            Dictionary with train and validation losses, single float for val_loss,
            or None if no datasets available
        """
        self.model.eval()
        eval_results = {}

        # Evaluate on training set
        if self.train_dataset is not None:
            train_loss = self._evaluate_dataset(self.train_dataset, "train")
            eval_results["train_loss"] = train_loss

        # Evaluate on validation set
        if self.val_dataset is not None:
            val_loss = self._evaluate_dataset(self.val_dataset, "val")
            eval_results["val_loss"] = val_loss

        return self._format_eval_results(eval_results)

    def _evaluate_dataset(self, dataset: TextDataset, dataset_name: str) -> float:
        """Evaluate model on a specific dataset."""
        losses = []
        for _ in range(self.config.eval_iters):
            try:
                x, y = dataset.get_batch(
                    batch_size=self.config.batch_size,
                    device=self.config.get_device(),
                )

                device = "cuda" if self.config.fp16 else "cpu"
                with autocast(device, enabled=self.config.fp16):
                    logits, loss = self.model(x, y)

                losses.append(loss.item())

            except Exception as e:
                logger.warning(f"Error during {dataset_name} evaluation: {e}")
                continue

        return sum(losses) / len(losses) if losses else float("inf")

    def _format_eval_results(
        self, eval_results: Dict[str, float]
    ) -> Optional[Union[Dict[str, float], float]]:
        """Format evaluation results for backward compatibility."""
        # Return single float for backward compatibility if only val_dataset
        if len(eval_results) == 1 and "val_loss" in eval_results:
            return eval_results["val_loss"]

        # Return None if no datasets available
        if not eval_results:
            return None

        return eval_results

    def train(self) -> Optional[Dict[str, float]]:
        """Run the complete training loop.

        Supports both epoch-based and iteration-based training modes.
        Handles evaluation, checkpointing, and progress logging.

        Returns:
            Final evaluation results
        """
        logger.info("Starting training...")
        start_time = time.time()

        # Determine training mode: iterations or epochs
        max_epochs = getattr(self.config, "max_epochs", None)
        max_iters = getattr(self.config, "max_iters", 10000)

        try:
            if max_epochs is not None:
                final_results = self._train_epochs(max_epochs, start_time)
            else:
                final_results = self._train_iterations(max_iters, start_time)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            final_results = self.evaluate()

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Final evaluation and logging
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Final results: {final_results}")

        return self._format_final_results(final_results)

    def _train_epochs(self, max_epochs: int, start_time: float) -> Optional[Union[Dict[str, float], float]]:
        """Train using epoch-based approach."""
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            avg_loss = self.train_epoch()

            # Evaluation and logging
            is_eval_epoch = (
                epoch % self.config.eval_interval == 0 or epoch == max_epochs - 1
            )
            if is_eval_epoch:
                eval_results = self.evaluate()
                self._log_epoch_progress(epoch, avg_loss, eval_results, start_time)
                self._save_best_model_if_improved(eval_results)

            # Save regular checkpoints
            self._save_regular_checkpoint_if_needed(epoch, max_epochs)

        return self.evaluate()

    def _train_iterations(self, max_iters: int, start_time: float) -> Optional[Union[Dict[str, float], float]]:
        """Train using iteration-based approach (legacy)."""
        for iteration in range(max_iters):
            self.current_iter = iteration
            self.global_step = iteration + 1  # Update global step for checkpoint naming
            self.train_step()

            # Save checkpoint every 100 steps
            if (iteration + 1) % 100 == 0:
                try:
                    checkpoint_path = self.save_checkpoint()
                    logger.info(f"Checkpoint {iteration + 1}: Saved to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {iteration + 1}: {e}")

            # Evaluation and logging
            is_eval_iter = (
                iteration % self.config.eval_interval == 0
                or iteration == max_iters - 1
            )
            if is_eval_iter:
                eval_results = self.evaluate()
                self._log_iteration_progress(iteration, eval_results, start_time)
                self._save_best_model_if_improved(eval_results)

        return self.evaluate()

    def _log_epoch_progress(
        self,
        epoch: int,
        avg_loss: float,
        eval_results: Optional[Union[Dict[str, float], float]],
        start_time: float,
    ) -> None:
        """Log progress for epoch-based training."""
        elapsed = time.time() - start_time
        train_loss, val_loss = self._extract_losses(eval_results, avg_loss)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    def _log_iteration_progress(
        self,
        iteration: int,
        eval_results: Optional[Union[Dict[str, float], float]],
        start_time: float,
    ) -> None:
        """Log progress for iteration-based training."""
        elapsed = time.time() - start_time
        train_loss, val_loss = self._extract_losses(eval_results, 0.0)

        logger.info(
            f"Step {iteration:5d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    def _extract_losses(
        self, eval_results: Optional[Union[Dict[str, float], float]], default_train: float
    ) -> Tuple[float, float]:
        """Extract train and validation losses from evaluation results."""
        if isinstance(eval_results, dict):
            train_loss = eval_results.get("train_loss", default_train)
            val_loss = eval_results.get("val_loss", 0.0)
        else:
            train_loss = default_train
            val_loss = eval_results if eval_results is not None else 0.0

        return train_loss, val_loss

    def _save_best_model_if_improved(
        self, eval_results: Optional[Union[Dict[str, float], float]]
    ) -> None:
        """Save model if validation loss improved."""
        if isinstance(eval_results, dict):
            val_loss = eval_results.get("val_loss", float("inf"))
        else:
            val_loss = eval_results if eval_results is not None else float("inf")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(self.save_dir / "best_model.pth")
            logger.info(f"New best model saved with val loss: {val_loss:.4f}")

    def _save_regular_checkpoint_if_needed(self, epoch: int, max_epochs: int) -> None:
        """Save regular checkpoints based on save_interval."""
        save_interval = getattr(self.config, "save_interval", self.config.eval_interval)
        if epoch % save_interval == 0 or epoch == max_epochs - 1:
            self.save_checkpoint()

    def _format_final_results(
        self, final_results: Optional[Union[Dict[str, float], float]]
    ) -> Optional[Dict[str, float]]:
        """Format final results for consistent return type."""
        if isinstance(final_results, dict):
            return final_results
        elif final_results is not None:
            return {"val_loss": final_results}
        else:
            return None

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss for the epoch

        Raises:
            RuntimeError: If no training dataset is provided
        """
        if self.train_dataset is None:
            raise RuntimeError("No training dataset provided")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Simple epoch training (could be enhanced with proper DataLoader)
        max_batches = min(
            self.config.max_batches_per_epoch, len(self.train_dataset)
        )

        for _ in range(max_batches):
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
        """Save model checkpoint with complete training state.

        Args:
            file_path: Path to save the checkpoint (optional, auto-generated if None)

        Returns:
            Path where checkpoint was saved

        Raises:
            Exception: If checkpoint saving fails
        """
        if file_path is None:
            file_path = self.save_dir / f"checkpoint_step_{self.global_step}.pth"
        else:
            file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            checkpoint = self._create_checkpoint_dict()
            torch.save(checkpoint, file_path)
            logger.info(f"Saved checkpoint to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def _create_checkpoint_dict(self) -> Dict[str, Any]:
        """Create checkpoint dictionary with all necessary state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "current_iter": self.current_iter,
            "current_epoch": self.current_epoch,
            "epoch": self.current_epoch,  # Backward compatibility
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        # Add vocab size if available
        vocab_size = self._get_vocab_size_from_datasets()
        if vocab_size is not None:
            checkpoint["vocab_size"] = vocab_size

        return checkpoint

    def _get_vocab_size_from_datasets(self) -> Optional[int]:
        """Get vocabulary size from available datasets."""
        if self.train_dataset is not None:
            return self.train_dataset.tokenizer.vocab_size
        elif self.val_dataset is not None:
            return self.val_dataset.tokenizer.vocab_size
        return None

    def load_checkpoint(self, file_path: Union[str, Path]) -> None:
        """Load model checkpoint and restore training state.

        Args:
            file_path: Path to the checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            Exception: If checkpoint loading fails
        """
        file_path = Path(file_path)

        from ..utils.helpers import _check_file_existence
        _check_file_existence(file_path, "Checkpoint")

        try:
            checkpoint = torch.load(
                file_path, map_location=self.config.device, weights_only=False
            )

            self._load_checkpoint_state(checkpoint)

            logger.info(f"Checkpoint loaded from {file_path}")
            logger.info(
                f"Resumed from iteration {self.current_iter}, "
                f"epoch {self.current_epoch}"
            )

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _load_checkpoint_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load state from checkpoint dictionary."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint.get("current_iter", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt using the trained model.

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
        generation_params = self._get_generation_params(max_tokens, temperature, top_k)

        self.model.eval()

        try:
            # Encode prompt and generate
            prompt_tensor = self._encode_prompt(prompt, tokenizer)
            generated_ids = self._generate_tokens(prompt_tensor, generation_params)

            # Decode with error handling
            return self._decode_generated_text(generated_ids, tokenizer)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation failed: {e}"

    def _get_generation_params(
        self, max_tokens: Optional[int], temperature: Optional[float], top_k: Optional[int]
    ) -> Dict[str, Any]:
        """Get generation parameters with config defaults."""
        return {
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "temperature": temperature or self.config.default_temperature,
            "top_k": top_k or self.config.default_top_k,
        }

    def _encode_prompt(self, prompt: str, tokenizer: Any) -> torch.Tensor:
        """Encode prompt text to tensor."""
        encoded_prompt = tokenizer.encode(prompt)
        return torch.tensor(
            [encoded_prompt], dtype=torch.long, device=self.device
        )

    def _generate_tokens(self, prompt_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Generate tokens using model's generate method or fallback."""
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                return self.model.generate(
                    prompt_tensor,
                    max_new_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                )
            else:
                return self._fallback_generation(prompt_tensor, params)

    def _fallback_generation(self, prompt_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Fallback generation method when model doesn't have generate method."""
        generated_ids = prompt_tensor.clone()
        vocab_size = self.config.vocab_size

        for _ in range(params["max_tokens"]):
            device = "cuda" if self.config.fp16 else "cpu"
            with autocast(device, enabled=self.config.fp16):
                logits, _ = self.model(generated_ids)

            # Sample next token
            next_token = self._sample_next_token(logits, params, vocab_size)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids

    def _sample_next_token(self, logits: torch.Tensor, params: Dict[str, Any], vocab_size: int) -> torch.Tensor:
        """Sample next token from logits with temperature and top-k."""
        # Get last token logits and apply temperature
        logits = logits[:, -1, :] / params["temperature"]

        # Ensure we don't sample tokens outside vocab
        logits = logits[:, :vocab_size]

        # Apply top-k filtering
        if params["top_k"] is not None:
            v, _ = torch.topk(logits, min(params["top_k"], logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _decode_generated_text(self, generated_ids: torch.Tensor, tokenizer: Any) -> str:
        """Decode generated token IDs to text with error handling."""
        try:
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            return str(generated_text)
        except Exception:
            # Fallback: decode token by token, skipping problematic ones
            return self._decode_with_fallback(generated_ids[0].tolist(), tokenizer)

    def _decode_with_fallback(self, token_ids: list, tokenizer: Any) -> str:
        """Decode tokens with fallback for problematic tokens."""
        decoded_parts = []
        for token_id in token_ids:
            try:
                decoded_parts.append(tokenizer.decode([token_id]))
            except Exception:
                logger.warning(f"Skipping token ID {token_id} during decoding")
                continue
        return "".join(decoded_parts)

    def generate_sample(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Generate text sample from a prompt (legacy method - use generate_text instead).

        Args:
            prompt: Text prompt to start generation
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated text including the prompt

        Raises:
            ValueError: If no tokenizer is available from datasets
        """
        # Use config defaults if not specified
        generation_params = self._get_generation_params(max_new_tokens, temperature, top_k)

        # Get tokenizer from dataset
        tokenizer = self._get_tokenizer_from_datasets()
        if tokenizer is None:
            raise ValueError(
                "No tokenizer available - provide dataset or use "
                "generate_text with tokenizer"
            )

        return self.generate_text(
            prompt,
            tokenizer,
            generation_params["max_tokens"],
            generation_params["temperature"],
            generation_params["top_k"]
        )

    def _get_tokenizer_from_datasets(self) -> Optional[Any]:
        """Get tokenizer from available datasets."""
        if self.train_dataset is not None:
            return self.train_dataset.tokenizer
        elif self.val_dataset is not None:
            return self.val_dataset.tokenizer
        return None


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
