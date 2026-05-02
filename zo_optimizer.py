from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-6,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.perturbation_mode = perturbation_mode
        self.step_count = 0
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Layer names not found: {missing}")
        return {n: named[n] for n in self.layer_names}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        perturbations = {}
        with torch.no_grad():
            for name, param in params.items():
                delta = torch.randint(0, 2, param.shape,
                                      device=param.device).float() * 2 - 1
                perturbations[name] = delta

            for name, param in params.items():
                param.data.add_(self.eps * perturbations[name])
            f_plus = loss_fn()

            for name, param in params.items():
                param.data.sub_(2.0 * self.eps * perturbations[name])
            f_minus = loss_fn()

            for name, param in params.items():
                param.data.add_(self.eps * perturbations[name])

        grad_scalar = (f_plus - f_minus) / (2.0 * self.eps)
        return {name: grad_scalar * perturbations[name] for name in params}

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(self.lr * grads[name])

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        # Сохраняем копии весов перед шагом
        backup = {name: param.data.clone() for name, param in params.items()}

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        # Проверяем — если стало хуже, откатываем
        with torch.no_grad():
            loss_after = loss_fn()

        if loss_after > loss_before:
            with torch.no_grad():
                for name, param in params.items():
                    param.data.copy_(backup[name])

        return float(loss_before)
