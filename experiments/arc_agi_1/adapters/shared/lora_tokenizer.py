from __future__ import annotations

from collections import OrderedDict
from math import prod
from typing import Any

import torch
from torch import Tensor


class Tokenizer2DBatchedLoRA:
    """HY-WU-style LoRA tensor tokenizer with a leading batch dimension."""

    def __init__(
        self,
        token_dim: int,
        rank: int,
        alpha: int,
        pg_mapping: dict[str, dict[str, int]],
        padding_value: float = 0.0,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")

        self.token_size = (rank, token_dim)
        self.rank = rank
        self.alpha = alpha
        self.padding_value = padding_value
        self.pg_mapping = self._expand_mapping(pg_mapping)
        self._module_names = list(self.pg_mapping.keys())

    def _expand_mapping(self, pg_mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        expanded: dict[str, dict[str, Any]] = {}
        for module, dims in pg_mapping.items():
            if isinstance(dims, dict) and dims.get("type") == "grouped":
                num_experts = dims["num_experts"]
                for sub_name, sub_dims in dims["sub_weights"].items():
                    expanded[f"{module}.{sub_name}"] = {
                        **sub_dims,
                        "num_experts": num_experts,
                    }
            else:
                expanded[module] = dict(dims)
        return expanded

    def get_key_order(self, key: str) -> int:
        minor_order = None
        for idx, module_name in enumerate(self._module_names):
            if module_name in key:
                minor_order = idx
                break
        if minor_order is None:
            raise ValueError(f"Key {key} does not match any pg_mapping module.")

        if "lora_A" in key:
            major_order = 0
        elif "lora_B" in key:
            major_order = 1
        else:
            raise ValueError(f"Key {key} must contain lora_A or lora_B.")
        return major_order * len(self._module_names) + minor_order

    def sorted_items(self, state_dict: dict[str, Tensor]):
        return OrderedDict(sorted(state_dict.items(), key=lambda item: self.get_key_order(item[0]))).items()

    def tokenize(self, state_dict: dict[str, Tensor]) -> Tensor:
        token_area = self.rank * self.token_size[1]
        all_tokens: list[Tensor] = []

        for _, weight in self.sorted_items(state_dict):
            if weight.dim() < 2:
                raise ValueError("LoRA weights must include a leading batch dimension and at least one parameter dim.")
            batch_size = weight.shape[0]
            flat = weight.reshape(batch_size, -1)
            param_count = flat.shape[1]
            num_tokens = (param_count + token_area - 1) // token_area
            padded_length = num_tokens * token_area
            if param_count < padded_length:
                padding = torch.full(
                    (batch_size, padded_length - param_count),
                    self.padding_value,
                    dtype=flat.dtype,
                    device=flat.device,
                )
                flat = torch.cat([flat, padding], dim=1)
            all_tokens.append(flat.reshape(batch_size, num_tokens, self.rank, self.token_size[1]))

        if not all_tokens:
            raise ValueError("state_dict is empty.")
        return torch.cat(all_tokens, dim=1)

    def detokenize(self, shape_state_dict: dict[str, tuple[int, ...]], tokens: Tensor) -> OrderedDict[str, Tensor]:
        ordered_shapes = OrderedDict(sorted(shape_state_dict.items(), key=lambda item: self.get_key_order(item[0])))
        batch_size = tokens.shape[0]
        token_area = self.rank * self.token_size[1]
        token_start = 0
        out: OrderedDict[str, Tensor] = OrderedDict()

        for key, shape in ordered_shapes.items():
            param_count = prod(shape)
            num_tokens = (param_count + token_area - 1) // token_area
            token_end = token_start + num_tokens
            param_tokens = tokens[:, token_start:token_end]
            flat = param_tokens.reshape(batch_size, -1)[:, :param_count]
            out[key] = flat.reshape(batch_size, *shape)
            token_start = token_end

        if token_start != tokens.shape[1]:
            raise ValueError(f"Unused tokens during detokenize: used {token_start}, got {tokens.shape[1]}.")
        return out

    @property
    def lora_scale(self) -> float:
        return self.alpha / self.rank

    @property
    def shape_state_dict(self) -> OrderedDict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for module in self._module_names:
            dims = self.pg_mapping[module]
            if "num_experts" in dims:
                num_experts = int(dims["num_experts"])
                shapes[f"{module}.lora_A"] = (num_experts, self.rank, int(dims["lora_A_dim"]))
                shapes[f"{module}.lora_B"] = (num_experts, int(dims["lora_B_dim"]), self.rank)
            else:
                shapes[f"{module}.lora_A"] = (self.rank, int(dims["lora_A_dim"]))
                shapes[f"{module}.lora_B"] = (int(dims["lora_B_dim"]), self.rank)
        return OrderedDict(sorted(shapes.items(), key=lambda item: self.get_key_order(item[0])))

    def _count_tokens(self, needle: str) -> int:
        token_area = self.rank * self.token_size[1]
        count = 0
        for key, shape in self.shape_state_dict.items():
            if needle in key:
                count += (prod(shape) + token_area - 1) // token_area
        return count

    @property
    def lora_A_token_count(self) -> int:
        return self._count_tokens("lora_A")

    @property
    def lora_B_token_count(self) -> int:
        return self._count_tokens("lora_B")

    def __call__(self, *args: Any, **kwargs: Any) -> OrderedDict[str, Tensor]:
        return self.detokenize(*args, **kwargs)
