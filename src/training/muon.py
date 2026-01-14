"""Muon Optimizer Implementation.

Muon (MomentUm Orthogonalized by Newton-Schulz) is an optimizer specifically
designed for 2D parameters in neural network hidden layers. It combines
SGD-momentum with matrix orthogonalization via Newton-Schulz iteration.

Based on: https://kellerjordan.github.io/posts/muon/
Reference: https://github.com/KellerJordan/Muon

Key insight: Apply Muon to 2D hidden layer parameters only, use AdamW for
embeddings, output layers, biases, and normalization weights.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Apply Newton-Schulz iteration to orthogonalize a matrix.

    This function approximates the nearest semi-orthogonal matrix to G.
    The result satisfies UV^T where USV^T is the SVD of G, effectively
    normalizing all singular values to 1.

    Args:
        G: Input gradient matrix (2D tensor)
        steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small value for numerical stability

    Returns:
        Orthogonalized matrix of same shape as G
    """
    # Tuned coefficients for fast convergence
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in bfloat16 for efficiency (if available, else float32)
    original_dtype = G.dtype
    if G.device.type == 'cuda' and torch.cuda.is_bf16_supported():
        X = G.to(torch.bfloat16)
    else:
        X = G.to(torch.float32)

    # Normalize to unit norm
    X = X / (X.norm() + eps)

    # Transpose if tall matrix (more rows than columns)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = X.T

    return X.to(original_dtype)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Wrapper for Newton-Schulz orthogonalization.

    Computes the matrix with the same singular vectors as G but all
    singular values set to 1 (the "zero-th power" of G).

    Args:
        G: Input gradient matrix
        steps: Number of NS iterations

    Returns:
        Orthogonalized matrix
    """
    return newtonschulz5(G, steps=steps)


class Muon(Optimizer):
    """Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.

    This optimizer applies Newton-Schulz orthogonalization to gradient
    updates for 2D hidden layer parameters, while using AdamW for all
    other parameters (embeddings, output layer, biases, normalization).

    Args:
        muon_params: Parameters to optimize with Muon (2D hidden weights)
        adamw_params: Parameters to optimize with AdamW (embeddings, etc.)
        lr: Learning rate for Muon params (default: 0.02)
        adamw_lr: Learning rate for AdamW params (default: 3e-4)
        momentum: Momentum coefficient for Muon (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        adamw_betas: Betas for AdamW (default: (0.9, 0.95))
        adamw_eps: Epsilon for AdamW (default: 1e-8)
        weight_decay: Weight decay for AdamW params (default: 0.01)
    """

    def __init__(
        self,
        muon_params: List[torch.nn.Parameter],
        adamw_params: List[torch.nn.Parameter],
        lr: float = 0.02,
        adamw_lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        # Store parameters
        self.muon_params = list(muon_params)
        self.adamw_params = list(adamw_params)
        self.lr = lr
        self.adamw_lr = adamw_lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adamw_betas = adamw_betas
        self.adamw_eps = adamw_eps
        self.weight_decay = weight_decay

        # Initialize state for Muon params (momentum buffers)
        self.muon_state: Dict[torch.nn.Parameter, torch.Tensor] = {}

        # Initialize state for AdamW params (exp_avg and exp_avg_sq)
        self.adamw_state: Dict[torch.nn.Parameter, Dict[str, torch.Tensor]] = {}
        self.adamw_step = 0

        # Create param groups for compatibility
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

        # Combine all params for base optimizer
        all_params = self.muon_params + self.adamw_params
        super().__init__(all_params, defaults)

        # Log parameter counts
        muon_count = sum(p.numel() for p in self.muon_params)
        adamw_count = sum(p.numel() for p in self.adamw_params)
        logger.info(
            f"Muon optimizer initialized: "
            f"{len(self.muon_params)} Muon params ({muon_count:,} elements), "
            f"{len(self.adamw_params)} AdamW params ({adamw_count:,} elements)"
        )

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: Optional closure for loss computation

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update Muon params
        self._step_muon()

        # Update AdamW params
        self._step_adamw()

        return loss

    def _step_muon(self) -> None:
        """Update parameters using Muon optimizer."""
        for p in self.muon_params:
            if p.grad is None:
                continue

            grad = p.grad

            # Initialize momentum buffer if needed
            if p not in self.muon_state:
                self.muon_state[p] = torch.zeros_like(grad)

            buf = self.muon_state[p]

            # Apply momentum (Nesterov or standard)
            buf.mul_(self.momentum).add_(grad)

            if self.nesterov:
                # Nesterov: use grad + momentum * buf
                update = grad + self.momentum * buf
            else:
                update = buf

            # Apply Newton-Schulz orthogonalization
            update_orth = zeropower_via_newtonschulz5(update, steps=self.ns_steps)

            # Scale by sqrt(max(m, n)) for proper normalization
            scale = max(p.size(0), p.size(1)) ** 0.5

            # Update parameters
            p.add_(update_orth, alpha=-self.lr * scale)

    def _step_adamw(self) -> None:
        """Update parameters using AdamW optimizer."""
        self.adamw_step += 1

        beta1, beta2 = self.adamw_betas

        for p in self.adamw_params:
            if p.grad is None:
                continue

            grad = p.grad

            # Initialize state if needed
            if p not in self.adamw_state:
                self.adamw_state[p] = {
                    'exp_avg': torch.zeros_like(p),
                    'exp_avg_sq': torch.zeros_like(p),
                }

            state = self.adamw_state[p]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            # Decoupled weight decay
            p.mul_(1 - self.adamw_lr * self.weight_decay)

            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Update biased second moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** self.adamw_step
            bias_correction2 = 1 - beta2 ** self.adamw_step

            # Compute step
            step_size = self.adamw_lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(self.adamw_eps)

            # Update parameters
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of all parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zeroing
        """
        for p in self.muon_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

        for p in self.adamw_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


def create_muon_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    ns_steps: int = 5,
) -> Muon:
    """Create a Muon optimizer with proper parameter grouping.

    Automatically categorizes model parameters:
    - Muon: 2D weight matrices in hidden layers (attention, MLP)
    - AdamW: Embeddings, output layer, biases, normalization weights

    Args:
        model: PyTorch model to optimize
        muon_lr: Learning rate for Muon params
        adamw_lr: Learning rate for AdamW params
        momentum: Momentum for Muon
        weight_decay: Weight decay for AdamW params
        ns_steps: Newton-Schulz iteration steps

    Returns:
        Configured Muon optimizer
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if this is a Muon param (2D hidden layer weight)
        is_2d = param.ndim == 2
        is_embedding = 'embed' in name.lower() or 'wte' in name.lower() or 'wpe' in name.lower()
        is_output = 'lm_head' in name.lower() or 'output' in name.lower()
        is_norm = 'norm' in name.lower() or 'ln' in name.lower()
        is_bias = 'bias' in name.lower() or name.endswith('.bias')

        if is_2d and not is_embedding and not is_output and not is_norm and not is_bias:
            muon_params.append(param)
            logger.debug(f"Muon param: {name} {param.shape}")
        else:
            adamw_params.append(param)
            logger.debug(f"AdamW param: {name} {param.shape}")

    return Muon(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=muon_lr,
        adamw_lr=adamw_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )
