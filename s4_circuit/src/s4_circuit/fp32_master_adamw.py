from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


class FP32MasterAdamW(torch.optim.AdamW):
    """Opt-in AdamW with FP32 updates and unchanged low-precision forward weights.

    Clip/all-reduce gradients on the model before step(), as usual. The optimizer
    owns FP32 copies only for FP16/BF16 parameters. Its checkpoint also stores
    those unrounded weights; loading a legacy optimizer checkpoint is rejected.
    No existing experiment selects this optimizer implicitly. The caller uses
    BF16 without GradScaler; scaled FP16 gradients are not supported here.
    """

    FORMAT = "s4_fp32_master_adamw_v1"

    def __init__(self, params: Iterable, **kwargs: Any) -> None:
        if kwargs.get("differentiable") or kwargs.get("capturable"):
            raise ValueError("FP32 master AdamW does not support differentiable/capturable mode")
        entries = list(params)
        if not entries:
            raise ValueError("FP32 master AdamW requires parameters")
        if isinstance(entries[0], dict):
            groups = [dict(group) for group in entries]
        else:
            groups = [{"params": entries}]
        self._model_groups: list[list[Tensor]] = []
        self._groups_locked = False
        seen: set[int] = set()
        for group in groups:
            originals = list(group["params"])
            masters = []
            for parameter in originals:
                if id(parameter) in seen:
                    raise ValueError("FP32 master AdamW received a duplicate parameter")
                seen.add(id(parameter))
                if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError("FP32 master AdamW requires FP16/BF16/FP32 parameters")
                if not parameter.is_leaf:
                    raise ValueError("FP32 master AdamW requires leaf parameters")
                master = parameter if parameter.dtype == torch.float32 else (
                    torch.nn.Parameter(parameter.detach().float().clone())
                )
                masters.append(master)
            self._model_groups.append(originals)
            group["params"] = masters
        super().__init__(groups, **kwargs)
        self._groups_locked = True

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._groups_locked:
            raise ValueError("Create FP32 master AdamW with all parameter groups up front")
        super().add_param_group(param_group)

    def _pairs(self):
        for originals, group in zip(self._model_groups, self.param_groups, strict=True):
            yield from zip(originals, group["params"], strict=True)

    @torch.no_grad()
    def step(self, closure=None):
        if hasattr(self, "grad_scale") or hasattr(self, "found_inf"):
            raise ValueError("FP32 master AdamW requires unscaled gradients without GradScaler")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for model, master in self._pairs():
            if master is model:
                continue
            gradient = model.grad
            if gradient is not None and gradient.is_sparse:
                raise RuntimeError("FP32 master AdamW does not support sparse gradients")
            master.grad = None if gradient is None else gradient.detach().float()
        super().step()
        for model, master in self._pairs():
            if model is not master and master.grad is not None:
                model.copy_(master)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for model, master in self._pairs():
            if model is master or model.grad is None:
                continue
            if set_to_none:
                model.grad = None
            else:
                model.grad.detach_()
                model.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["fp32_master"] = {
            "format": self.FORMAT,
            "model_dtypes": [[str(p.dtype) for p in group] for group in self._model_groups],
            "weights": [
                [None if model is master else master.detach().clone()
                 for model, master in zip(originals, group["params"], strict=True)]
                for originals, group in zip(self._model_groups, self.param_groups, strict=True)
            ],
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        payload = state_dict.get("fp32_master", {})
        if payload.get("format") != self.FORMAT:
            raise ValueError("Optimizer checkpoint has no compatible FP32 master state")
        expected_dtypes = [[str(p.dtype) for p in group] for group in self._model_groups]
        if payload.get("model_dtypes") != expected_dtypes:
            raise ValueError("FP32 master checkpoint model dtypes/groups changed")
        weights = payload.get("weights", [])
        if len(weights) != len(self.param_groups):
            raise ValueError("FP32 master checkpoint group count changed")
        pairs = []
        for originals, group, saved in zip(self._model_groups, self.param_groups, weights, strict=True):
            if len(saved) != len(originals):
                raise ValueError("FP32 master checkpoint parameter count changed")
            for model, master, weight in zip(originals, group["params"], saved, strict=True):
                if model is master:
                    if weight is not None:
                        raise ValueError("FP32 forward parameter unexpectedly has a saved master")
                elif (not isinstance(weight, Tensor) or weight.dtype != torch.float32
                      or weight.shape != master.shape):
                    raise ValueError("FP32 master checkpoint tensor shape/dtype changed")
                elif not torch.isfinite(weight).all():
                    raise ValueError("FP32 master checkpoint weights must be finite")
                pairs.append((model, master, weight))
        super().load_state_dict({key: value for key, value in state_dict.items() if key != "fp32_master"})
        with torch.no_grad():
            for model, master, weight in pairs:
                if weight is not None:
                    master.copy_(weight)
                    model.copy_(master)

    def precision_summary(self) -> dict[str, Any]:
        low_precision_count = sum(p.numel() for p, m in self._pairs() if p is not m)
        return {
            "format": self.FORMAT,
            "low_precision_parameter_count": low_precision_count,
            "fp32_master_weight_bytes": 4 * low_precision_count,
            # Existing BF16 AdamW has 4 bytes/element of moments; FP32 has 8.
            "extra_persistent_bytes_vs_bf16_adamw": 8 * low_precision_count,
            "fp32_gradient_copy_bytes_at_step": 4 * low_precision_count,
        }
