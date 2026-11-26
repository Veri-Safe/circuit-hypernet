from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch import Tensor

PARITY_FORMAT = "nvarc_qwen3_hf_tl_parity_v1"


@dataclass(frozen=True)
class ParityTolerance:
    atol: float
    rtol: float

    def validate(self) -> None:
        if self.atol < 0.0 or self.rtol < 0.0:
            raise ValueError("Parity tolerances must be nonnegative")


def compare_tensor(
    left: Tensor,
    right: Tensor,
    *,
    label: str,
    tolerance: ParityTolerance,
) -> dict[str, float | list[int] | str]:
    tolerance.validate()
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {left.shape} != {right.shape}")
    left_value = left.detach().float().cpu()
    right_value = right.detach().float().cpu()
    difference = (left_value - right_value).abs()
    denominator = torch.maximum(left_value.abs(), right_value.abs()).clamp_min(1e-12)
    report: dict[str, float | list[int] | str] = {
        "label": label,
        "shape": list(left.shape),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "max_rel": (
            float((difference / denominator).max().item())
            if difference.numel()
            else 0.0
        ),
    }
    if not torch.allclose(
        left_value, right_value, atol=tolerance.atol, rtol=tolerance.rtol
    ):
        raise ValueError(f"HF/TL parity failed for {label}: {report}")
    return report


def _bfloat16_downward_ulp(value: Tensor) -> float:
    value_bf16 = torch.tensor(float(value.float().item()), dtype=torch.bfloat16)
    lower = torch.nextafter(
        value_bf16,
        torch.tensor(float("-inf"), dtype=torch.bfloat16),
    )
    return float((value_bf16.float() - lower.float()).item())


def _capture_hf_mlp_layers(
    model: Any, input_ids: Tensor
) -> tuple[Tensor, dict[int, Tensor], dict[int, Tensor]]:
    inputs: dict[int, Tensor] = {}
    outputs: dict[int, Tensor] = {}
    handles = []
    for layer, block in enumerate(model.model.layers):

        def pre_hook(
            _module: Any,
            arguments: tuple[Any, ...],
            *,
            layer_index: int = layer,
        ) -> None:
            inputs[layer_index] = arguments[0].detach().cpu()

        def post_hook(
            _module: Any,
            _arguments: tuple[Any, ...],
            output: Tensor,
            *,
            layer_index: int = layer,
        ) -> None:
            outputs[layer_index] = output.detach().cpu()

        handles.extend(
            (
                block.mlp.register_forward_pre_hook(pre_hook),
                block.mlp.register_forward_hook(post_hook),
            )
        )
    try:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    expected = set(range(36))
    if set(inputs) != expected or set(outputs) != expected:
        raise RuntimeError("HF MLP hooks did not fire for all 36 layers")
    return result.logits.detach().cpu(), inputs, outputs


def run_hf_tl_parity(
    *,
    hf_model: Any,
    tl_model: Any,
    token_ids: Sequence[int],
    logit_tolerance: ParityTolerance,
    activation_tolerance: ParityTolerance,
) -> dict[str, Any]:
    if not token_ids:
        raise ValueError("Parity token sequence is empty")
    hf_device = next(hf_model.parameters()).device
    tl_device = torch.device(tl_model.cfg.device)
    hf_ids = torch.tensor([token_ids], dtype=torch.long, device=hf_device)
    tl_ids = torch.tensor([token_ids], dtype=torch.long, device=tl_device)
    hf_logits, hf_inputs, hf_outputs = _capture_hf_mlp_layers(hf_model, hf_ids)
    names = {
        hook
        for layer in range(36)
        for hook in (
            f"blocks.{layer}.mlp.hook_in",
            f"blocks.{layer}.mlp.hook_out",
        )
    }
    with torch.inference_mode():
        current_tl_logits, cache = tl_model.run_with_cache(
            tl_ids, names_filter=lambda name: name in names
        )
    tl_logits = current_tl_logits.detach().cpu()
    reports = [
        compare_tensor(
            hf_logits,
            tl_logits,
            label="logits",
            tolerance=logit_tolerance,
        )
    ]
    for layer in range(36):
        reports.append(
            compare_tensor(
                hf_inputs[layer],
                cache[f"blocks.{layer}.mlp.hook_in"],
                label=f"layer_{layer:02d}.mlp_input",
                tolerance=activation_tolerance,
            )
        )
        reports.append(
            compare_tensor(
                hf_outputs[layer],
                cache[f"blocks.{layer}.mlp.hook_out"],
                label=f"layer_{layer:02d}.mlp_output",
                tolerance=activation_tolerance,
            )
        )
    del cache
    del current_tl_logits
    hf_argmax = hf_logits.argmax(dim=-1).cpu()
    tl_argmax = tl_logits.argmax(dim=-1).cpu()
    if not torch.equal(hf_argmax, tl_argmax):
        raise ValueError("HF/TL argmax token parity failed")
    return {
        "format": PARITY_FORMAT,
        "tokens": len(token_ids),
        "layers": 36,
        "argmax_exact": True,
        "reports": reports,
    }


def validate_parity_suite(
    reports: Mapping[str, Mapping[str, Any]], *, expected_events: int = 12
) -> None:
    if len(reports) != expected_events:
        raise ValueError(
            f"Parity suite has {len(reports)} events, expected {expected_events}"
        )
    for event_key, report in reports.items():
        if len(event_key) != 64 or report.get("format") != PARITY_FORMAT:
            raise ValueError("Parity suite contains an invalid event or report")
        if report.get("argmax_exact") is not True or report.get("layers") != 36:
            raise ValueError(f"Parity suite event failed its contract: {event_key}")


def validate_tl_frozen_greedy_replay(
    *,
    tl_model: Any,
    condition_token_ids: Sequence[int],
    generation: Any,
    token_contract: Any,
    grammar_factory: Callable[[], Any],
    atol: float = 1e-4,
    rtol: float = 1e-4,
    require_expected_greedy: bool = True,
    align_target_diagnostics_to_span_replay: bool = False,
) -> dict[str, Any]:
    """Replay a frozen legal trajectory and optionally require greedy choices."""

    replay = tuple(int(item) for item in condition_token_ids) + tuple(
        int(item) for item in generation.generated_token_ids
    )
    device = torch.device(tl_model.cfg.device)
    input_ids = torch.tensor([replay], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = tl_model(input_ids)[0].float()
    grammar = grammar_factory()
    expected = (*generation.generated_token_ids, token_contract.eos_token_id)
    probabilities = []
    entropies = []
    log_probabilities = []
    tie_breaks_accepted = 0
    bf16_logit_ambiguities_accepted = 0
    bf16_logit_ambiguity_max_gap = 0.0
    argmax_mismatches = 0
    argmax_mismatch_max_gap = 0.0
    allowed_argmax_tokens = []
    for step, expected_token in enumerate(expected):
        position = len(condition_token_ids) - 1 + step
        allowed = grammar.allowed_token_ids(
            token_contract.digit_token_ids,
            token_contract.newline_token_id,
            token_contract.eos_token_id,
        )
        if not allowed:
            raise ValueError("TL parity grammar reached a dead end")
        row = logits[position]
        allowed_ids = torch.tensor(allowed, dtype=torch.long, device=row.device)
        allowed_logits = row.index_select(0, allowed_ids)
        best_allowed_index = int(allowed_logits.argmax().item())
        selected = int(allowed_ids[best_allowed_index].item())
        allowed_argmax_tokens.append(selected)
        expected_token = int(expected_token)
        expected_matches_allowed = allowed_ids == expected_token
        if not bool(expected_matches_allowed.any().item()):
            raise ValueError(
                f"TL constrained-greedy parity failed at step {step}: "
                f"{expected_token} is not grammar-allowed"
            )
        expected_index = int(expected_matches_allowed.nonzero()[0].item())
        expected_logit = allowed_logits[expected_index]
        best_logit = allowed_logits[best_allowed_index]
        expected_is_best_by_declared_tolerance = torch.isclose(
            expected_logit.float(),
            best_logit.float(),
            atol=atol,
            rtol=rtol,
        ) or bool((expected_logit >= best_logit).item())
        logit_gap = float((best_logit.float() - expected_logit.float()).item())
        bf16_logit_ambiguous = False
        if not expected_is_best_by_declared_tolerance and logit_gap > 0.0:
            bf16_tolerance = _bfloat16_downward_ulp(best_logit) + atol + (
                rtol * abs(float(best_logit.float().item()))
            )
            bf16_logit_ambiguous = logit_gap <= bf16_tolerance
        expected_is_best = (
            expected_is_best_by_declared_tolerance or bf16_logit_ambiguous
        )
        if selected != expected_token:
            argmax_mismatches += 1
            argmax_mismatch_max_gap = max(argmax_mismatch_max_gap, logit_gap)
        if not expected_is_best and require_expected_greedy:
            raise ValueError(
                f"TL constrained-greedy parity failed at step {step}: "
                f"{selected} != {expected_token}; "
                f"best_allowed_logit={float(best_logit.float().item())}, "
                f"expected_logit={float(expected_logit.float().item())}"
            )
        if selected != expected_token and expected_is_best:
            tie_breaks_accepted += 1
            if bf16_logit_ambiguous:
                bf16_logit_ambiguities_accepted += 1
                bf16_logit_ambiguity_max_gap = max(
                    bf16_logit_ambiguity_max_gap,
                    logit_gap,
                )
        log_probs = torch.log_softmax(row, dim=-1)
        probs = torch.exp(log_probs)
        probabilities.append(float(probs[expected_token].item()))
        entropies.append(float((-(probs * log_probs).sum()).item()))
        log_probabilities.append(float(log_probs[expected_token].item()))
        grammar.consume(
            expected_token,
            token_contract.digit_token_ids,
            token_contract.newline_token_id,
            token_contract.eos_token_id,
        )
    if not grammar.finished:
        raise ValueError("TL replay did not terminate its grid grammar")
    if align_target_diagnostics_to_span_replay:
        target_replay = tuple(int(item) for item in condition_token_ids) + tuple(
            int(item) for item in generation.generated_token_ids[:-1]
        )
        target_input_ids = torch.tensor(
            [target_replay], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            target_logits = tl_model(target_input_ids)[0].float()
        target_positions = torch.arange(
            len(condition_token_ids) - 1,
            len(condition_token_ids) - 1 + generation.target_count,
            dtype=torch.long,
            device=device,
        )
        target_ids = torch.tensor(
            generation.generated_token_ids, dtype=torch.long, device=device
        )
        target_rows = target_logits.index_select(0, target_positions)
        target_log_probs = torch.log_softmax(target_rows, dim=-1)
        target_probs = torch.exp(target_log_probs)
        selected_target_log_probs = target_log_probs.gather(
            1, target_ids[:, None]
        ).squeeze(1)
        selected_target_probs = torch.exp(selected_target_log_probs)
        target_entropies = -(target_probs * target_log_probs).sum(dim=-1)
        probabilities[: generation.target_count] = (
            selected_target_probs.detach().cpu().tolist()
        )
        entropies[: generation.target_count] = (
            target_entropies.detach().cpu().tolist()
        )
        log_probabilities[: generation.target_count] = (
            selected_target_log_probs.detach().cpu().tolist()
        )
    expected_probabilities = torch.tensor(
        (*generation.target_probabilities, generation.eos_probability),
        dtype=torch.float32,
    )
    actual_probabilities = torch.tensor(probabilities, dtype=torch.float32)
    probability_delta = (actual_probabilities - expected_probabilities).abs()
    probability_match = torch.allclose(
        actual_probabilities,
        expected_probabilities,
        atol=atol,
        rtol=rtol,
    )
    denominator = torch.maximum(
        actual_probabilities.abs(), expected_probabilities.abs()
    ).clamp_min(1e-12)
    return {
        "tokens_exact": True,
        "frozen_trajectory_tokens_preserved": True,
        "content_tokens": generation.target_count,
        "eos_exact": True,
        "expected_greedy_required": require_expected_greedy,
        "target_diagnostic_replay_shape": (
            "condition_plus_target_without_last_v1"
            if align_target_diagnostics_to_span_replay
            else "condition_plus_full_target_v1"
        ),
        "argmax_exact": argmax_mismatches == 0,
        "argmax_mismatches": argmax_mismatches,
        "argmax_mismatch_max_gap": argmax_mismatch_max_gap,
        "full_replay_allowed_argmax_sha256": hashlib.sha256(
            bytes(allowed_argmax_tokens)
        ).hexdigest(),
        "tie_breaks_accepted": tie_breaks_accepted,
        "bf16_logit_ambiguities_accepted": bf16_logit_ambiguities_accepted,
        "bf16_logit_ambiguity_max_gap": bf16_logit_ambiguity_max_gap,
        "full_softmax_selected_probabilities": probabilities,
        "target_probabilities": probabilities[: generation.target_count],
        "target_entropies": entropies[: generation.target_count],
        "target_log_probabilities": log_probabilities[: generation.target_count],
        "eos_probability": probabilities[generation.target_count],
        "eos_log_probability": log_probabilities[generation.target_count],
        "probabilities_match_native_hf": bool(probability_match),
        "native_hf_probability_max_abs": (
            float(probability_delta.max().item()) if probability_delta.numel() else 0.0
        ),
        "native_hf_probability_mean_abs": (
            float(probability_delta.mean().item()) if probability_delta.numel() else 0.0
        ),
        "native_hf_probability_max_rel": (
            float((probability_delta / denominator).max().item())
            if probability_delta.numel()
            else 0.0
        ),
    }
