from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from s2_hypernet.lora_tokenizer import Tokenizer2DBatchedLoRA
from s2_hypernet.nvarc_pg.target_layout import QwenTargetLayout


@dataclass(frozen=True)
class ParameterTokenRouting:
    module_names: tuple[str, ...]
    token_to_module: Tensor
    valid_slot_module_mask: Tensor

    def __post_init__(self) -> None:
        if self.token_to_module.dtype != torch.long or self.token_to_module.dim() != 1:
            raise ValueError("token_to_module must be a rank-1 int64 tensor")
        expected = (self.valid_slot_module_mask.shape[1],)
        if expected != (len(self.module_names),):
            raise ValueError("valid mask module dimension does not match module_names")
        if self.valid_slot_module_mask.dtype != torch.bool:
            raise ValueError("valid_slot_module_mask must be boolean")

    @property
    def internal_token_count(self) -> int:
        return int(self.token_to_module.numel())

    def token_valid_mask(self) -> Tensor:
        return self.valid_slot_module_mask[:, self.token_to_module]

    def indices_for_module(self, module_name: str) -> Tensor:
        try:
            module_index = self.module_names.index(module_name)
        except ValueError as error:
            raise KeyError(module_name) from error
        return torch.nonzero(self.token_to_module == module_index, as_tuple=False).flatten()


def build_parameter_token_routing(
    tokenizer: Tokenizer2DBatchedLoRA,
    *,
    layout: QwenTargetLayout,
) -> ParameterTokenRouting:
    module_names = tuple(module.name for module in layout.all_generator_modules)
    module_to_index = {name: index for index, name in enumerate(module_names)}
    token_area = tokenizer.rank * tokenizer.token_size[1]
    rank_after_accumulation = layout.rank // layout.dim_accumulation
    route: list[int] = []

    for key, shape in tokenizer.shape_state_dict.items():
        module_name, _, factor = key.rpartition(".lora_")
        if not module_name or factor not in {"A", "B"}:
            raise ValueError(f"Unexpected tokenizer key: {key!r}")
        try:
            module_index = module_to_index[module_name]
        except KeyError as error:
            raise ValueError(f"Tokenizer module is absent from Qwen layout: {module_name}") from error
        parameter_count = math.prod(shape)
        group_count = math.ceil(parameter_count / token_area)
        route.extend([module_index] * group_count * rank_after_accumulation)

    token_to_module = torch.tensor(route, dtype=torch.long)
    if token_to_module.numel() != layout.internal_parameter_tokens_per_slot:
        raise ValueError(
            "Parameter-token router length changed: "
            f"{token_to_module.numel()} != {layout.internal_parameter_tokens_per_slot}"
        )

    valid = torch.zeros(
        layout.generator_slot_count,
        len(module_names),
        dtype=torch.bool,
    )
    valid[: layout.num_layers, : len(layout.modules)] = True
    for global_offset, _ in enumerate(layout.global_modules):
        valid[layout.num_layers + global_offset, len(layout.modules) + global_offset] = True

    if int(valid.sum().item()) != len(layout.target_keys):
        raise ValueError("Valid slot/module mask does not contain exactly 253 Qwen targets")
    return ParameterTokenRouting(
        module_names=module_names,
        token_to_module=token_to_module,
        valid_slot_module_mask=valid,
    )

