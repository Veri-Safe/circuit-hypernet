from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .traces import TraceRecord


class _StopAfterMLP(RuntimeError):
    pass


def qwen_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Expected Hugging Face Qwen layers at model.model.layers")


@dataclass(frozen=True)
class CapturedActivations:
    mlp_input: Tensor
    mlp_output: Tensor
    eligible_mask: Tensor
    assistant_target_mask: Tensor


class LayerActivationCapture:
    """Capture one Qwen MLP and stop the frozen forward immediately afterwards."""

    def __init__(self, model: Any, layer: int) -> None:
        layers = qwen_layers(model)
        if not 0 <= layer < len(layers):
            raise ValueError(f"Layer {layer} is outside 0..{len(layers) - 1}")
        self.model = model
        self.layer = int(layer)
        self.mlp = layers[layer].mlp
        self._input: Tensor | None = None
        self._output: Tensor | None = None

    def _pre_hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        self._input = args[0].detach()

    def _post_hook(self, _module: Any, _args: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self._output = tensor.detach()
        raise _StopAfterMLP()

    def capture(self, record: TraceRecord, *, device: torch.device) -> CapturedActivations:
        record.validate()
        self._input = None
        self._output = None
        input_ids = torch.tensor(record.input_ids, device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        handles = (
            self.mlp.register_forward_pre_hook(self._pre_hook),
            self.mlp.register_forward_hook(self._post_hook),
        )
        try:
            with torch.inference_mode():
                try:
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                except _StopAfterMLP:
                    pass
        finally:
            for handle in handles:
                handle.remove()
        if self._input is None or self._output is None:
            raise RuntimeError(f"Failed to capture Qwen MLP layer {self.layer}")
        if self._input.shape != self._output.shape or self._input.shape[:2] != input_ids.shape:
            raise ValueError("Captured MLP activations do not align with trace tokens")
        return CapturedActivations(
            mlp_input=self._input[0],
            mlp_output=self._output[0],
            eligible_mask=torch.tensor(record.eligible_mask, device=device, dtype=torch.bool),
            assistant_target_mask=torch.tensor(
                record.assistant_target_mask, device=device, dtype=torch.bool
            ),
        )


def load_frozen_step399(
    model_path: str,
    tokenizer_path: str,
    *,
    device: torch.device,
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    if int(model.config.vocab_size) != 16 or int(model.config.num_hidden_layers) != 36:
        raise ValueError("Frozen model is not the locked 36-layer, 16-token Qwen")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    model.config.use_cache = False
    return model, tokenizer
