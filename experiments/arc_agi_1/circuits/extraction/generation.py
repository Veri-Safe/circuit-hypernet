from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .events import CircuitEvent
from .storage import load_safetensors_bundle, write_safetensors_bundle

GENERATION_FORMAT = "nvarc_rearc_frozen_generation_chunk_v1"
LEGACY_REPLAY_VALIDATION = "legacy_tl_frozen_greedy_v1"
FULL_REPLAY_GREEDY_VALIDATION = "native_hf_full_replay_greedy_v1"
S2_KV_FROZEN_TRAJECTORY_VALIDATION = "s2_kv_cache_frozen_trajectory_v1"
SUPPORTED_REPLAY_VALIDATION_MODES = frozenset(
    {
        LEGACY_REPLAY_VALIDATION,
        FULL_REPLAY_GREEDY_VALIDATION,
        S2_KV_FROZEN_TRAJECTORY_VALIDATION,
    }
)


def normalize_replay_validation_modes(value: Any) -> tuple[str, ...]:
    if value is None:
        return (LEGACY_REPLAY_VALIDATION,)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Replay validation modes must be a nonempty list")
    modes = tuple(str(item) for item in value)
    if len(set(modes)) != len(modes):
        raise ValueError("Replay validation modes contain duplicates")
    unsupported = sorted(set(modes) - SUPPORTED_REPLAY_VALIDATION_MODES)
    if unsupported:
        raise ValueError(f"Unsupported replay validation modes: {unsupported}")
    return modes


def merge_replay_validation_mode(value: Any, current: str) -> list[str]:
    if current not in SUPPORTED_REPLAY_VALIDATION_MODES:
        raise ValueError(f"Unsupported replay validation mode: {current}")
    modes = list(normalize_replay_validation_modes(value))
    if current not in modes:
        modes.append(current)
    return modes


class GridGrammar(Protocol):
    finished: bool

    def allowed_token_ids(
        self,
        digit_token_ids: Sequence[int],
        newline_token_id: int,
        eos_token_id: int,
    ) -> Sequence[int]: ...

    def consume(
        self,
        token_id: int,
        digit_token_ids: Sequence[int],
        newline_token_id: int,
        eos_token_id: int,
    ) -> None: ...


@dataclass(frozen=True)
class TokenContract:
    digit_token_ids: tuple[int, ...]
    newline_token_id: int
    pad_token_id: int
    eos_token_id: int
    vocab_size: int

    def validate(self) -> None:
        if self.digit_token_ids != tuple(range(10)):
            raise ValueError("NVARC digit token IDs changed")
        if (
            self.newline_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.vocab_size,
        ) != (
            10,
            13,
            15,
            16,
        ):
            raise ValueError("NVARC 16-token grammar contract changed")


@dataclass(frozen=True)
class GenerationRecord:
    event_key: str
    generated_token_ids: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_probabilities: tuple[float, ...]
    target_entropies: tuple[float, ...]
    target_log_probabilities: tuple[float, ...]
    eos_probability: float
    eos_log_probability: float
    finished: bool

    @property
    def target_count(self) -> int:
        return len(self.generated_token_ids)

    def validate(self, *, condition_length: int | None = None) -> None:
        if len(self.event_key) != 64:
            raise ValueError("Generation event key must be a SHA256")
        lengths = {
            self.target_count,
            len(self.target_positions),
            len(self.target_probabilities),
            len(self.target_entropies),
            len(self.target_log_probabilities),
        }
        if len(lengths) != 1 or not self.generated_token_ids:
            raise ValueError("Generation target arrays are empty or misaligned")
        if self.target_count > 929:
            raise ValueError("Generated content exceeds the legal 30x30 grid bound")
        if not self.finished:
            raise ValueError("Generation record did not terminate with EOS")
        if any(token_id not in range(11) for token_id in self.generated_token_ids):
            raise ValueError("Generation content contains a non-grid token")
        values = (
            *self.target_probabilities,
            *self.target_entropies,
            *self.target_log_probabilities,
            self.eos_probability,
            self.eos_log_probability,
        )
        if not torch.isfinite(torch.tensor(values, dtype=torch.float64)).all():
            raise ValueError("Generation diagnostics contain non-finite values")
        if any(not 0.0 <= value <= 1.0 for value in self.target_probabilities):
            raise ValueError("Target probability lies outside [0, 1]")
        if not 0.0 <= self.eos_probability <= 1.0:
            raise ValueError("EOS probability lies outside [0, 1]")
        if any(value < 0.0 for value in self.target_entropies):
            raise ValueError("Target entropy is negative")
        if (
            any(value > 1e-6 for value in self.target_log_probabilities)
            or self.eos_log_probability > 1e-6
        ):
            raise ValueError("A saved log probability is positive")
        if condition_length is not None:
            expected = tuple(
                range(condition_length - 1, condition_length + self.target_count - 1)
            )
            if self.target_positions != expected:
                raise ValueError(
                    f"Target positions changed: {self.target_positions} != {expected}"
                )


def verify_token_contract(tokenizer: Any, model: Any) -> TokenContract:
    for value in range(10):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if encoded != [value]:
            raise ValueError(f"Tokenizer digit contract changed for {value}: {encoded}")
    if tokenizer.encode("\n", add_special_tokens=False) != [10]:
        raise ValueError("Tokenizer newline contract changed")
    if tokenizer.convert_tokens_to_ids("<|im_start|>") != 14:
        raise ValueError("Tokenizer im_start contract changed")
    if tokenizer.convert_tokens_to_ids("<|im_end|>") != 15:
        raise ValueError("Tokenizer im_end contract changed")
    contract = TokenContract(
        digit_token_ids=tuple(range(10)),
        newline_token_id=10,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        vocab_size=int(model.config.vocab_size),
    )
    contract.validate()
    return contract


def load_native_hf_model_and_tokenizer(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    device: torch.device,
    attention_backend: str = "flash_attention_2",
) -> tuple[Any, Any, TokenContract]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), use_fast=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_backend,
        local_files_only=True,
    )
    if float(getattr(model.config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Frozen Qwen must have zero attention dropout")
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, tokenizer, verify_token_contract(tokenizer, model)


def _default_grammar_factory() -> GridGrammar:
    from nvarc_qwen3.nvarc_grid_grammar import GridGrammarState

    return GridGrammarState()


def _left_pad(
    conditions: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not conditions or any(not condition for condition in conditions):
        raise ValueError("Generation conditions must be nonempty")
    max_length = max(len(condition) for condition in conditions)
    input_ids = torch.full(
        (len(conditions), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, condition in enumerate(conditions):
        values = torch.tensor(condition, dtype=torch.long, device=device)
        input_ids[index, -len(condition) :] = values
        attention_mask[index, -len(condition) :] = 1
    return input_ids, attention_mask


def constrained_greedy_pretokenized_batch(
    model: Any,
    conditions: Sequence[Sequence[int]],
    *,
    event_keys: Sequence[str],
    token_contract: TokenContract,
    max_new_tokens: int,
    max_sequence_length: int,
    device: torch.device,
    grammar_factory: Callable[[], GridGrammar] = _default_grammar_factory,
) -> tuple[GenerationRecord, ...]:
    if len(conditions) != len(event_keys) or not conditions:
        raise ValueError("Generation conditions and event keys must align")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    token_contract.validate()
    if any(len(condition) > max_sequence_length for condition in conditions):
        raise ValueError("A generation condition exceeds the context limit")

    input_ids, attention_mask = _left_pad(
        conditions,
        pad_token_id=token_contract.pad_token_id,
        device=device,
    )
    grammars = [grammar_factory() for _ in conditions]
    tokens: list[list[int]] = [[] for _ in conditions]
    probabilities: list[list[float]] = [[] for _ in conditions]
    entropies: list[list[float]] = [[] for _ in conditions]
    log_probabilities: list[list[float]] = [[] for _ in conditions]
    eos_stats: list[tuple[float, float] | None] = [None] * len(conditions)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens):
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1)
            active_before = [not grammar.finished for grammar in grammars]
            allowed_mask = torch.zeros_like(log_probs, dtype=torch.bool)
            for batch_index, grammar in enumerate(grammars):
                if grammar.finished:
                    allowed_mask[batch_index, token_contract.pad_token_id] = True
                    continue
                allowed = grammar.allowed_token_ids(
                    token_contract.digit_token_ids,
                    token_contract.newline_token_id,
                    token_contract.eos_token_id,
                )
                if not allowed:
                    raise RuntimeError(f"Grid grammar reached a dead end: {grammar}")
                allowed_mask[batch_index, torch.tensor(allowed, device=device)] = True
            selected = torch.argmax(
                log_probs.masked_fill(~allowed_mask, float("-inf")), dim=-1
            )
            selected_log_probs = log_probs.gather(1, selected[:, None]).squeeze(1)
            selected_probs = probs.gather(1, selected[:, None]).squeeze(1)
            rows = (
                torch.stack(
                    (
                        selected.float(),
                        selected_probs,
                        selected_log_probs,
                        entropy,
                    ),
                    dim=1,
                )
                .cpu()
                .tolist()
            )
            next_tokens = [int(row[0]) for row in rows]
            for batch_index, grammar in enumerate(grammars):
                if grammar.finished:
                    continue
                token_id = next_tokens[batch_index]
                probability, log_probability, token_entropy = map(
                    float,
                    (rows[batch_index][1], rows[batch_index][2], rows[batch_index][3]),
                )
                grammar.consume(
                    token_id,
                    token_contract.digit_token_ids,
                    token_contract.newline_token_id,
                    token_contract.eos_token_id,
                )
                if token_id == token_contract.eos_token_id:
                    eos_stats[batch_index] = (probability, log_probability)
                else:
                    tokens[batch_index].append(token_id)
                    probabilities[batch_index].append(probability)
                    log_probabilities[batch_index].append(log_probability)
                    entropies[batch_index].append(token_entropy)
            if all(grammar.finished for grammar in grammars):
                break
            for batch_index, active in enumerate(active_before):
                if active and not grammars[batch_index].finished:
                    replay_length = len(conditions[batch_index]) + len(
                        tokens[batch_index]
                    )
                    if replay_length > max_sequence_length:
                        raise RuntimeError(
                            f"Event {event_keys[batch_index]} exceeded context without EOS"
                        )
            next_token_tensor = torch.tensor(
                next_tokens, dtype=torch.long, device=device
            )[:, None]
            new_mask = torch.tensor(
                active_before, dtype=attention_mask.dtype, device=device
            )[:, None]
            attention_mask = torch.cat((attention_mask, new_mask), dim=1)
            outputs = model(
                input_ids=next_token_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

    records = []
    for index, grammar in enumerate(grammars):
        if not grammar.finished or eos_stats[index] is None:
            raise RuntimeError(
                f"Event {event_keys[index]} did not terminate with EOS within "
                f"max_new_tokens={max_new_tokens}"
            )
        if not tokens[index]:
            raise RuntimeError(f"Event {event_keys[index]} generated an empty grid")
        eos_probability, eos_log_probability = eos_stats[index]
        condition_length = len(conditions[index])
        record = GenerationRecord(
            event_key=event_keys[index],
            generated_token_ids=tuple(tokens[index]),
            target_positions=tuple(
                range(condition_length - 1, condition_length + len(tokens[index]) - 1)
            ),
            target_probabilities=tuple(probabilities[index]),
            target_entropies=tuple(entropies[index]),
            target_log_probabilities=tuple(log_probabilities[index]),
            eos_probability=eos_probability,
            eos_log_probability=eos_log_probability,
            finished=True,
        )
        record.validate(condition_length=condition_length)
        records.append(record)
    return tuple(records)


def generate_events(
    model: Any,
    events: Sequence[CircuitEvent],
    *,
    token_contract: TokenContract,
    max_new_tokens: int,
    max_sequence_length: int,
    device: torch.device,
    grammar_factory: Callable[[], GridGrammar] = _default_grammar_factory,
) -> tuple[GenerationRecord, ...]:
    records = constrained_greedy_pretokenized_batch(
        model,
        [event.condition_input_ids for event in events],
        event_keys=[event.event_key for event in events],
        token_contract=token_contract,
        max_new_tokens=max_new_tokens,
        max_sequence_length=max_sequence_length,
        device=device,
        grammar_factory=grammar_factory,
    )
    for event, record in zip(events, records, strict=True):
        record.validate(condition_length=event.condition_length)
    return records


def recompute_generation_replay_diagnostics(
    model: Any,
    conditions: Sequence[Sequence[int]],
    records: Sequence[GenerationRecord],
    *,
    token_contract: TokenContract,
    device: torch.device,
    max_sequence_length: int,
    grammar_factory: Callable[[], GridGrammar] = _default_grammar_factory,
    require_greedy_exact: bool = False,
    replay_audit_sink: MutableSequence[dict[str, Any]] | None = None,
) -> tuple[GenerationRecord, ...]:
    """Recompute saved probabilities from an unpadded full causal replay.

    Token selection remains frozen from ``records``. This pass makes the stored
    diagnostics describe the same replay shape consumed by exact attribution.
    """

    if len(conditions) != len(records) or not records:
        raise ValueError("Replay conditions and generation records must align")
    token_contract.validate()
    result = []
    for condition, record in zip(conditions, records, strict=True):
        record.validate(condition_length=len(condition))
        replay = tuple(int(item) for item in condition) + tuple(
            int(item) for item in record.generated_token_ids
        )
        if len(replay) > max_sequence_length:
            raise ValueError(
                f"Replay for {record.event_key} exceeds max_sequence_length"
            )
        input_ids = torch.tensor([replay], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[0].float()
        expected = (*record.generated_token_ids, token_contract.eos_token_id)
        positions = tuple(range(len(condition) - 1, len(condition) + len(expected) - 1))
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        selected_tensor = torch.tensor(expected, dtype=torch.long, device=device)
        selected_logits = logits.index_select(0, position_tensor)
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        probabilities = torch.exp(log_probs)
        selected_log_probs = log_probs.gather(1, selected_tensor[:, None]).squeeze(1)
        selected_probabilities = torch.exp(selected_log_probs)
        entropies = -(probabilities * log_probs).sum(dim=-1)

        grammar = grammar_factory()
        greedy_mismatches = 0
        exact_tie_mismatches = 0
        max_logit_gap = 0.0
        first_mismatch = None
        full_replay_argmax_tokens = []
        for step, expected_token in enumerate(expected):
            allowed = grammar.allowed_token_ids(
                token_contract.digit_token_ids,
                token_contract.newline_token_id,
                token_contract.eos_token_id,
            )
            if not allowed:
                raise ValueError("HF replay grammar reached a dead end")
            if int(expected_token) not in allowed:
                raise ValueError(
                    f"Frozen replay token violates the grid grammar: {expected_token}"
                )
            allowed_ids = torch.tensor(
                allowed, dtype=torch.long, device=selected_logits.device
            )
            allowed_logits = selected_logits[step].index_select(0, allowed_ids)
            best_allowed_index = int(allowed_logits.argmax().item())
            selected_token = int(allowed_ids[best_allowed_index].item())
            full_replay_argmax_tokens.append(selected_token)
            expected_matches = allowed_ids == int(expected_token)
            expected_index = int(expected_matches.nonzero()[0].item())
            best_logit = float(allowed_logits[best_allowed_index].item())
            expected_logit = float(allowed_logits[expected_index].item())
            logit_gap = max(0.0, best_logit - expected_logit)
            if selected_token != int(expected_token):
                greedy_mismatches += 1
                max_logit_gap = max(max_logit_gap, logit_gap)
                if logit_gap == 0.0:
                    exact_tie_mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "step": step,
                        "cached_token_id": int(expected_token),
                        "full_replay_token_id": selected_token,
                        "best_allowed_logit": best_logit,
                        "cached_token_logit": expected_logit,
                        "logit_gap": logit_gap,
                    }
                if require_greedy_exact:
                    raise ValueError(
                        "Native-HF full-replay constrained-greedy parity failed "
                        f"at step {step} for event {record.event_key}: "
                        f"{selected_token} != {int(expected_token)}; "
                        f"best_allowed_logit={best_logit}, "
                        f"expected_logit={expected_logit}"
                    )
            grammar.consume(
                int(expected_token),
                token_contract.digit_token_ids,
                token_contract.newline_token_id,
                token_contract.eos_token_id,
            )
        if not grammar.finished:
            raise ValueError("HF replay did not terminate its grid grammar")
        if replay_audit_sink is not None:
            replay_audit_sink.append(
                {
                    "format": "nvarc_native_hf_cache_full_replay_audit_v1",
                    "event_key": record.event_key,
                    "checked_steps": len(expected),
                    "argmax_exact": greedy_mismatches == 0,
                    "argmax_mismatches": greedy_mismatches,
                    "exact_tie_mismatches": exact_tie_mismatches,
                    "max_logit_gap": max_logit_gap,
                    "full_replay_allowed_argmax_sha256": hashlib.sha256(
                        bytes(full_replay_argmax_tokens)
                    ).hexdigest(),
                    "first_mismatch": first_mismatch,
                }
            )

        content_count = record.target_count
        replay_record = GenerationRecord(
            event_key=record.event_key,
            generated_token_ids=record.generated_token_ids,
            target_positions=record.target_positions,
            target_probabilities=tuple(
                float(item) for item in selected_probabilities[:content_count].tolist()
            ),
            target_entropies=tuple(
                float(item) for item in entropies[:content_count].tolist()
            ),
            target_log_probabilities=tuple(
                float(item) for item in selected_log_probs[:content_count].tolist()
            ),
            eos_probability=float(selected_probabilities[content_count].item()),
            eos_log_probability=float(selected_log_probs[content_count].item()),
            finished=True,
        )
        replay_record.validate(condition_length=len(condition))
        result.append(replay_record)
    return tuple(result)


def write_generation_chunk(
    path: str | Path,
    records: Sequence[GenerationRecord],
    *,
    config_sha256: str,
    model_identity_sha256: str,
    tokenizer_identity_sha256: str,
    batch_size: int,
    full_replay_greedy_required: bool = False,
    replay_validation_mode: str = LEGACY_REPLAY_VALIDATION,
    full_replay_greedy_audits: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot write an empty generation chunk")
    if replay_validation_mode not in SUPPORTED_REPLAY_VALIDATION_MODES:
        raise ValueError(
            f"Unsupported replay validation mode: {replay_validation_mode}"
        )
    if full_replay_greedy_audits is not None and len(
        full_replay_greedy_audits
    ) != len(records):
        raise ValueError("Full-replay audits must align one-to-one with records")
    if (
        replay_validation_mode != LEGACY_REPLAY_VALIDATION
        and full_replay_greedy_audits is None
    ):
        raise ValueError("Non-legacy replay validation requires per-event audits")
    if full_replay_greedy_audits is not None:
        audit_event_keys = tuple(
            str(value.get("event_key", "")) for value in full_replay_greedy_audits
        )
        record_event_keys = tuple(record.event_key for record in records)
        if audit_event_keys != record_event_keys:
            raise ValueError("Full-replay audit event order changed")
    offsets = [0]
    token_ids: list[int] = []
    positions: list[int] = []
    probabilities: list[float] = []
    entropies: list[float] = []
    log_probabilities: list[float] = []
    metadata_records = []
    for record in records:
        record.validate()
        token_ids.extend(record.generated_token_ids)
        positions.extend(record.target_positions)
        probabilities.extend(record.target_probabilities)
        entropies.extend(record.target_entropies)
        log_probabilities.extend(record.target_log_probabilities)
        offsets.append(len(token_ids))
        value = asdict(record)
        for key in (
            "generated_token_ids",
            "target_positions",
            "target_probabilities",
            "target_entropies",
            "target_log_probabilities",
        ):
            value.pop(key)
        metadata_records.append(value)
    return write_safetensors_bundle(
        path,
        tensors={
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "generated_token_ids": torch.tensor(token_ids, dtype=torch.int16),
            "target_positions": torch.tensor(positions, dtype=torch.int32),
            "target_probabilities": torch.tensor(probabilities, dtype=torch.float32),
            "target_entropies": torch.tensor(entropies, dtype=torch.float32),
            "target_log_probabilities": torch.tensor(
                log_probabilities, dtype=torch.float32
            ),
        },
        metadata={
            "format": GENERATION_FORMAT,
            "config_sha256": config_sha256,
            "model_identity_sha256": model_identity_sha256,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
            "decoder": "constrained_greedy_full_softmax_v1",
            "diagnostic_source": "native_hf_full_causal_replay_v1",
            "batch_size": int(batch_size),
            "full_replay_greedy_required": bool(full_replay_greedy_required),
            "replay_validation_mode": replay_validation_mode,
            "full_replay_greedy_audits": (
                [dict(value) for value in full_replay_greedy_audits]
                if full_replay_greedy_audits is not None
                else []
            ),
            "events": len(records),
            "records": metadata_records,
        },
    )


def read_generation_chunk(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    metadata_sink: MutableSequence[dict[str, Any]] | None = None,
) -> tuple[GenerationRecord, ...]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=GENERATION_FORMAT)
    if metadata_sink is not None:
        metadata_sink.append(dict(metadata))
    if expected_identity is not None:
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_identity.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Generation chunk identity changed: {mismatches}")
    records = metadata.get("records")
    offsets = tensors["offsets"].tolist()
    if not isinstance(records, list) or len(offsets) != len(records) + 1:
        raise ValueError("Generation records do not align with offsets")
    result = []
    for index, value in enumerate(records):
        start, end = offsets[index : index + 2]
        record = GenerationRecord(
            **value,
            generated_token_ids=tuple(
                int(item) for item in tensors["generated_token_ids"][start:end].tolist()
            ),
            target_positions=tuple(
                int(item) for item in tensors["target_positions"][start:end].tolist()
            ),
            target_probabilities=tuple(
                float(item)
                for item in tensors["target_probabilities"][start:end].tolist()
            ),
            target_entropies=tuple(
                float(item) for item in tensors["target_entropies"][start:end].tolist()
            ),
            target_log_probabilities=tuple(
                float(item)
                for item in tensors["target_log_probabilities"][start:end].tolist()
            ),
        )
        record.validate()
        result.append(record)
    return tuple(result)
