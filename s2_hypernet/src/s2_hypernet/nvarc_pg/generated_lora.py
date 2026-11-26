from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn


def batched_lora_delta(hidden: Tensor, lora_a: Tensor, lora_b: Tensor) -> Tensor:
    """Apply one LoRA factor pair per leading-batch item."""

    squeezed = False
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(1)
        squeezed = True
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden rank 2 or 3, found {tuple(hidden.shape)}")
    if lora_a.dim() != 3 or lora_b.dim() != 3:
        raise ValueError("Generated LoRA factors must have a leading batch dimension")
    if hidden.shape[0] != lora_a.shape[0] or lora_a.shape[0] != lora_b.shape[0]:
        raise ValueError("Each query in the hidden batch requires exactly one LoRA")
    if hidden.shape[-1] != lora_a.shape[-1]:
        raise ValueError("Generated LoRA A input dimension does not match module input")
    if lora_a.shape[1] != lora_b.shape[-1]:
        raise ValueError("Generated LoRA A/B rank mismatch")

    lora_a = lora_a.to(device=hidden.device, dtype=hidden.dtype)
    lora_b = lora_b.to(device=hidden.device, dtype=hidden.dtype)
    delta = torch.matmul(
        torch.matmul(hidden, lora_a.transpose(1, 2)),
        lora_b.transpose(1, 2),
    )
    return delta.squeeze(1) if squeezed else delta


class GeneratedLoRAController:
    """Install scoped Generated LoRA hooks without mutating model weights."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_keys: tuple[str, ...],
        scale_multiplier: float = 1.0,
    ) -> None:
        if not target_keys:
            raise ValueError("target_keys must not be empty")
        self.model = model
        self.target_keys = tuple(target_keys)
        self._target_scope = "all"
        self._enabled_target_keys = frozenset(self.target_keys)
        self._scale_multiplier = 1.0
        self._target_scale_multipliers: dict[str, float | Tensor] = {}
        self._state: Mapping[str, Tensor] | None = None
        self.set_scale_multiplier(scale_multiplier)
        self._active_keys: set[str] = set()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._install()

    def _install(self) -> None:
        for target_key in self.target_keys:
            try:
                module = self.model.get_submodule(target_key)
            except AttributeError as error:
                raise ValueError(f"Generated LoRA target does not exist: {target_key}") from error
            if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
                raise TypeError(f"Generated LoRA target is not linear-like: {target_key}")

            def hook(
                hooked_module: nn.Module,
                inputs: tuple[Tensor, ...],
                output: Tensor,
                *,
                key: str = target_key,
            ) -> Tensor:
                del hooked_module
                state = self._state
                if state is None or key not in self._enabled_target_keys:
                    return output
                if not inputs:
                    raise TypeError(f"Generated LoRA target {key} received no positional input")
                hidden = inputs[0]
                lora_a = state[f"{key}.lora_A"]
                lora_b = state[f"{key}.lora_B"]
                scale = state["scale"].to(device=hidden.device, dtype=hidden.dtype)
                scale = scale * self._scale_multiplier
                target_multiplier = self._target_scale_multipliers.get(key)
                if target_multiplier is not None:
                    if isinstance(target_multiplier, Tensor):
                        target_multiplier = target_multiplier.to(
                            device=hidden.device, dtype=hidden.dtype
                        )
                    scale = scale * target_multiplier
                self._active_keys.add(key)
                return output + batched_lora_delta(hidden, lora_a, lora_b) * scale

            self._handles.append(module.register_forward_hook(hook))

    @property
    def scale_multiplier(self) -> float:
        return self._scale_multiplier

    def set_scale_multiplier(self, value: float) -> None:
        multiplier = float(value)
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                "Generated LoRA scale multiplier must be finite and non-negative"
            )
        if self._state is not None:
            raise RuntimeError("Cannot change the scale multiplier while LoRA is active")
        self._scale_multiplier = multiplier

    @property
    def target_scale_multipliers(self) -> dict[str, float | Tensor]:
        """Return the explicit nonidentity or differentiable target coefficients."""

        return dict(self._target_scale_multipliers)

    def set_target_scale_multipliers(
        self, values: Mapping[str, float | Tensor] | None
    ) -> None:
        """Replace optional multipliers on each target's effective ``s * B @ A``.

        The state already contains any checkpoint gate. These multipliers are
        applied exactly once after the existing global multiplier; neither
        factor is changed. Omitted targets use 1. Float and nondifferentiable
        tensor identities retain the original arithmetic. A scalar tensor with
        gradients stays in the graph even at 1, for training-source diagnostics.
        Callers must not mutate coefficient tensors while an activation is live.
        """

        if self._state is not None:
            raise RuntimeError("Cannot change target scales while LoRA is active")
        values = {} if values is None else values
        unknown = set(values) - set(self.target_keys)
        if unknown:
            raise ValueError(f"Unknown Generated LoRA targets: {sorted(unknown)}")
        validated: dict[str, float | Tensor] = {}
        for key, value in values.items():
            if isinstance(value, Tensor):
                if value.ndim != 0 or not value.is_floating_point():
                    raise ValueError("Target scale tensors must be floating scalars")
                numeric = float(value.detach().item())
                stored: float | Tensor = value
            else:
                numeric = float(value)
                stored = numeric
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError("Target scales must be finite and non-negative")
            if numeric != 1.0 or (isinstance(value, Tensor) and value.requires_grad):
                validated[key] = stored
        self._target_scale_multipliers = validated

    @contextmanager
    def target_scale_context(
        self, values: Mapping[str, float | Tensor] | None
    ) -> Iterator["GeneratedLoRAController"]:
        """Temporarily replace target scales, restoring them after activation."""

        previous = self.target_scale_multipliers
        self.set_target_scale_multipliers(values)
        try:
            yield self
        finally:
            self.set_target_scale_multipliers(previous)

    @property
    def target_scope(self) -> str:
        return self._target_scope

    def set_target_scope(self, scope: str) -> None:
        """Select which generated LoRA targets affect evaluation forwards.

        This is deliberately an evaluation-time residual mask: the Parameter
        Generator still emits its complete state, while hooks outside the
        selected scope return the frozen module output unchanged.
        """

        if self._state is not None:
            raise RuntimeError("Cannot change the target scope while LoRA is active")
        normalized = str(scope).strip().lower()
        if normalized == "all":
            enabled = frozenset(self.target_keys)
        elif normalized == "qv":
            enabled = frozenset(
                key
                for key in self.target_keys
                if key == "self_attn.q_proj"
                or key == "self_attn.v_proj"
                or key.endswith(".self_attn.q_proj")
                or key.endswith(".self_attn.v_proj")
            )
            if not enabled:
                raise ValueError(
                    "Generated LoRA target layout contains no q/v projections"
                )
        else:
            raise ValueError("Generated LoRA target scope must be one of: all, qv")
        self._target_scope = normalized
        self._enabled_target_keys = enabled

    def _validate_state(self, state: Mapping[str, Tensor]) -> None:
        missing = []
        for target_key in self.target_keys:
            for factor in ("lora_A", "lora_B"):
                key = f"{target_key}.{factor}"
                if key not in state:
                    missing.append(key)
        if "scale" not in state:
            missing.append("scale")
        if missing:
            raise ValueError(f"Generated LoRA state is missing keys: {missing[:10]}")

    @contextmanager
    def activate(self, state: Mapping[str, Tensor]) -> Iterator["GeneratedLoRAController"]:
        """Keep one state active for the entire forward and backward pair."""

        if self._state is not None:
            raise RuntimeError("Generated LoRA activation contexts cannot be nested")
        self._validate_state(state)
        self._state = state
        self._active_keys = set()
        try:
            yield self
        finally:
            self._state = None

    def activation_summary(self) -> dict[str, object]:
        inactive = sorted(self._enabled_target_keys - self._active_keys)
        disabled = sorted(set(self.target_keys) - self._enabled_target_keys)
        return {
            "active": len(self._active_keys),
            "total": len(self._enabled_target_keys),
            "inactive": inactive,
            "disabled": disabled,
            "scale_multiplier": self._scale_multiplier,
            "target_scope": self._target_scope,
        }

    def close(self) -> None:
        if self._state is not None:
            raise RuntimeError("Cannot remove Generated LoRA hooks while a state is active")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def slice_generated_lora_state(
    state: Mapping[str, Tensor],
    index: int,
) -> dict[str, Tensor]:
    """Select one query while preserving scalar control tensors."""

    result: dict[str, Tensor] = {}
    for key, value in state.items():
        if key == "scale" or value.dim() == 0:
            result[key] = value
        else:
            if not 0 <= index < value.shape[0]:
                raise IndexError(f"Generated LoRA batch index {index} is outside {value.shape[0]}")
            result[key] = value[index : index + 1]
    return result
