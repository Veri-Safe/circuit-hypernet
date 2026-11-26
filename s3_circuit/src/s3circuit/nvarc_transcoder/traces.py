from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TraceEvent:
    event_index: int
    message_index: int
    response_start: int
    valid_answer_end: int
    response_end: int
    generated_token_ids: tuple[int, ...]

    @property
    def anchor_index(self) -> int:
        if self.valid_answer_end <= self.response_start:
            raise ValueError("Trace event has no valid answer token")
        return self.valid_answer_end - 1


@dataclass(frozen=True)
class TraceRecord:
    dataset_index: int
    puzzle_name: str | None
    input_ids: tuple[int, ...]
    eligible_mask: tuple[bool, ...]
    assistant_target_mask: tuple[bool, ...]
    events: tuple[TraceEvent, ...]

    def validate(self, *, max_sequence_length: int = 16_384) -> None:
        length = len(self.input_ids)
        if not 0 < length <= max_sequence_length:
            raise ValueError(f"Invalid trace length: {length}")
        if len(self.eligible_mask) != length or len(self.assistant_target_mask) != length:
            raise ValueError("Trace masks do not align with input_ids")
        if not self.events:
            raise ValueError("Trace has no assistant prediction events")
        for event in self.events:
            if not (
                0 <= event.response_start
                < event.valid_answer_end
                <= event.response_end
                <= length
            ):
                raise ValueError(f"Invalid trace event boundary: {event}")
            if not self.assistant_target_mask[event.anchor_index]:
                raise ValueError("Event anchor is not an assistant target token")


def trace_record_sha256(record: TraceRecord) -> str:
    """Return a stable digest of every field that affects trace training."""

    record.validate()
    payload = {
        "dataset_index": record.dataset_index,
        "puzzle_name": record.puzzle_name,
        "input_ids": record.input_ids,
        "eligible_mask": record.eligible_mask,
        "assistant_target_mask": record.assistant_target_mask,
        "events": tuple(asdict(event) for event in record.events),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _metadata_path(tensor_path: Path) -> Path:
    return tensor_path.with_suffix(".json")


def _completion_path(tensor_path: Path) -> Path:
    return Path(str(tensor_path) + ".done")


def _trace_tensor_path(artifact: Path) -> Path | None:
    name = artifact.name
    for suffix in (
        ".safetensors.done.tmp",
        ".safetensors.done",
        ".safetensors.tmp",
        ".safetensors",
        ".json.tmp",
        ".json",
    ):
        if name.startswith("chunk-") and name.endswith(suffix):
            stem = name[: -len(suffix)]
            return artifact.with_name(stem + ".safetensors")
    return None


def recover_incomplete_trace_artifacts(shard_dir: str | Path) -> tuple[str, ...]:
    """Remove only uncommitted chunk artifacts left by an interrupted write.

    A valid ``.done`` marker is the commit point. If it exists, both payloads
    must exist; otherwise the chunk is corrupt and requires operator review.
    Without a valid marker, all files for that chunk are safe to regenerate.
    """

    root = Path(shard_dir)
    tensor_paths = {
        tensor_path
        for artifact in root.glob("chunk-*")
        if (tensor_path := _trace_tensor_path(artifact)) is not None
    }
    removed: list[str] = []
    for tensor_path in sorted(tensor_paths):
        metadata_path = _metadata_path(tensor_path)
        completion_path = _completion_path(tensor_path)
        marker_is_valid = (
            completion_path.is_file()
            and completion_path.read_bytes() == b"ok\n"
        )
        if marker_is_valid:
            if not tensor_path.is_file() or not metadata_path.is_file():
                raise ValueError(
                    "Committed trace chunk is missing its tensor or metadata payload: "
                    f"{tensor_path}"
                )
            continue
        for artifact in (
            tensor_path,
            metadata_path,
            tensor_path.with_name(tensor_path.name + ".tmp"),
            metadata_path.with_name(metadata_path.name + ".tmp"),
            completion_path,
            completion_path.with_name(completion_path.name + ".tmp"),
        ):
            if artifact.is_file():
                artifact.unlink()
                removed.append(str(artifact))
    return tuple(removed)


def write_trace_file(path: str | Path, records: Sequence[TraceRecord]) -> dict[str, Any]:
    from safetensors.torch import save_file

    if not records:
        raise ValueError("Cannot write an empty trace file")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        record.validate()

    offsets = [0]
    input_ids: list[int] = []
    eligible: list[int] = []
    assistant: list[int] = []
    metadata = []
    for record in records:
        input_ids.extend(record.input_ids)
        eligible.extend(int(value) for value in record.eligible_mask)
        assistant.extend(int(value) for value in record.assistant_target_mask)
        offsets.append(len(input_ids))
        metadata.append(
            {
                "dataset_index": record.dataset_index,
                "puzzle_name": record.puzzle_name,
                "length": len(record.input_ids),
                "events": [asdict(event) for event in record.events],
            }
        )
    tensors = {
        "input_ids": torch.tensor(input_ids, dtype=torch.int16),
        "eligible_mask": torch.tensor(eligible, dtype=torch.uint8),
        "assistant_target_mask": torch.tensor(assistant, dtype=torch.uint8),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    save_file(tensors, str(temporary))
    metadata_destination = _metadata_path(destination)
    metadata_temporary = metadata_destination.with_name(metadata_destination.name + ".tmp")
    metadata_temporary.write_text(
        json.dumps(
            {"format": "nvarc_generated_trace_shard_v1", "records": metadata},
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    os.replace(metadata_temporary, metadata_destination)
    completion_destination = _completion_path(destination)
    completion_temporary = completion_destination.with_name(
        completion_destination.name + ".tmp"
    )
    completion_temporary.write_bytes(b"ok\n")
    os.replace(completion_temporary, completion_destination)
    return {
        "path": str(destination.resolve()),
        "records": len(records),
        "tokens": len(input_ids),
        "events": sum(len(record.events) for record in records),
        "first_dataset_index": records[0].dataset_index,
        "last_dataset_index": records[-1].dataset_index,
    }


def read_trace_file(path: str | Path) -> tuple[TraceRecord, ...]:
    from safetensors.torch import load_file

    source = Path(path)
    completion_path = _completion_path(source)
    if (
        not completion_path.is_file()
        or completion_path.read_bytes() != b"ok\n"
    ):
        raise ValueError(f"Trace file is not complete: {source}")
    metadata = json.loads(_metadata_path(source).read_text(encoding="utf-8"))
    if metadata.get("format") != "nvarc_generated_trace_shard_v1":
        raise ValueError(f"Unsupported trace metadata: {source}")
    tensors = load_file(str(source), device="cpu")
    offsets = tensors["offsets"].tolist()
    rows = metadata["records"]
    if len(offsets) != len(rows) + 1:
        raise ValueError("Trace offsets do not match metadata rows")
    result = []
    for index, row in enumerate(rows):
        start, end = int(offsets[index]), int(offsets[index + 1])
        events = tuple(
            TraceEvent(
                event_index=int(event["event_index"]),
                message_index=int(event["message_index"]),
                response_start=int(event["response_start"]),
                valid_answer_end=int(event["valid_answer_end"]),
                response_end=int(event["response_end"]),
                generated_token_ids=tuple(int(item) for item in event["generated_token_ids"]),
            )
            for event in row["events"]
        )
        record = TraceRecord(
            dataset_index=int(row["dataset_index"]),
            puzzle_name=row.get("puzzle_name"),
            input_ids=tuple(int(item) for item in tensors["input_ids"][start:end].tolist()),
            eligible_mask=tuple(
                bool(item) for item in tensors["eligible_mask"][start:end].tolist()
            ),
            assistant_target_mask=tuple(
                bool(item)
                for item in tensors["assistant_target_mask"][start:end].tolist()
            ),
            events=events,
        )
        record.validate()
        result.append(record)
    return tuple(result)


def trace_files(root: str | Path) -> tuple[Path, ...]:
    resolved = Path(root)
    return tuple(
        path
        for path in sorted(resolved.glob("shard-*-of-*/chunk-*.safetensors"))
        if _completion_path(path).is_file()
    )


def iter_trace_records(root: str | Path) -> Iterator[TraceRecord]:
    previous = -1
    for path in trace_files(root):
        for record in read_trace_file(path):
            if record.dataset_index <= previous:
                raise ValueError("Trace root is not in strictly increasing dataset order")
            previous = record.dataset_index
            yield record


def inspect_trace_root(
    root: str | Path,
    *,
    expected_rows: int,
    expected_events: int,
) -> dict[str, Any]:
    files = trace_files(root)
    if not files:
        raise FileNotFoundError(f"No complete trace shards under {Path(root).resolve()}")
    rows = 0
    events = 0
    tokens = 0
    eligible_tokens = 0
    expected_index = 0
    for record in iter_trace_records(root):
        if record.dataset_index != expected_index:
            raise ValueError(
                f"Trace coverage gap: expected dataset index {expected_index}, "
                f"found {record.dataset_index}"
            )
        expected_index += 1
        rows += 1
        events += len(record.events)
        tokens += len(record.input_ids)
        eligible_tokens += sum(record.eligible_mask)
    if rows != expected_rows or events != expected_events:
        raise ValueError(
            f"Trace identity mismatch: rows={rows}, events={events}, "
            f"expected={expected_rows}/{expected_events}"
        )
    return {
        "format": "nvarc_generated_trace_root_v1",
        "root": str(Path(root).resolve()),
        "files": len(files),
        "rows": rows,
        "events": events,
        "tokens": tokens,
        "eligible_tokens": eligible_tokens,
    }


def load_trace_manifest(
    root: str | Path,
    *,
    expected_rows: int,
    expected_events: int,
    expected_base_identity_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(root).resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Finalized trace manifest is missing: {path}; run finalize_nvarc_traces.py"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": "nvarc_generated_trace_root_v1",
        "rows": int(expected_rows),
        "events": int(expected_events),
        "status": "complete",
    }
    if expected_base_identity_sha256 is not None:
        expected["base_identity_sha256"] = expected_base_identity_sha256
    if expected_config_sha256 is not None:
        expected["config_sha256"] = expected_config_sha256
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Finalized trace manifest mismatch: {mismatches}")
    return value


def verify_token_contract(tokenizer: Any, model: Any) -> dict[str, Any]:
    digit_ids = []
    for value in range(10):
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if encoded != [value]:
            raise ValueError(f"16-token digit contract changed for {value}: {encoded}")
        digit_ids.append(value)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if newline != [10]:
        raise ValueError(f"16-token newline contract changed: {newline}")
    if tokenizer.pad_token_id != 13 or tokenizer.eos_token_id != 15:
        raise ValueError("16-token pad/eos contract changed")
    if int(model.config.vocab_size) != 16:
        raise ValueError("Frozen Qwen model does not have a 16-token vocabulary")
    return {
        "digit_token_ids": tuple(digit_ids),
        "newline_token_id": 10,
        "pad_token_id": 13,
        "eos_token_id": 15,
    }


def _first_differing_index(left: str, right: str) -> int:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    return min(len(left), len(right))


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(item) for item in encoded]


def _decode_answer(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def constrained_greedy_generate(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[int, ...]:
    from nvarc_qwen3.nvarc_grid_grammar import GridGrammarState

    prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    grammar = GridGrammarState()
    generated: list[int] = []
    digit_ids = tuple(int(item) for item in token_contract["digit_token_ids"])
    newline_id = int(token_contract["newline_token_id"])
    eos_id = int(token_contract["eos_token_id"])
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
        for _ in range(max_new_tokens):
            allowed = grammar.allowed_token_ids(digit_ids, newline_id, eos_id)
            if not allowed:
                raise RuntimeError(f"Grid grammar reached a dead end: {grammar}")
            allowed_tensor = torch.tensor(allowed, device=device, dtype=torch.long)
            selected = int(
                allowed[
                    int(torch.argmax(logits[0].index_select(0, allowed_tensor)).item())
                ]
            )
            generated.append(selected)
            grammar.consume(selected, digit_ids, newline_id, eos_id)
            if grammar.finished:
                break
            next_token = torch.tensor([[selected]], device=device, dtype=torch.long)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
    if not grammar.finished or not generated or generated[-1] != eos_id:
        raise RuntimeError("Constrained greedy generation did not terminate with EOS")
    return tuple(generated)


def build_generated_trace(
    row: Mapping[str, Any],
    *,
    dataset_index: int,
    model: Any,
    tokenizer: Any,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    max_sequence_length: int,
    device: torch.device,
) -> TraceRecord:
    source_messages = row["messages"]
    if not source_messages or len(source_messages) % 2:
        raise ValueError("NVARC row must contain complete user/assistant pairs")
    generated_messages: list[dict[str, Any]] = []
    generated_answers: list[tuple[int, ...]] = []
    for message_index in range(0, len(source_messages), 2):
        user = source_messages[message_index]
        assistant = source_messages[message_index + 1]
        if str(user["role"]).lower() != "user" or str(assistant["role"]).lower() != "assistant":
            raise ValueError("NVARC row does not alternate user/assistant")
        generated_messages.append({"role": "user", "content": user["content"]})
        generated = constrained_greedy_generate(
            model,
            tokenizer,
            generated_messages,
            token_contract=token_contract,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        content_ids = generated[:-1]
        if not content_ids:
            raise RuntimeError("Generated grid has no answer token before EOS")
        generated_answers.append(tuple(content_ids))
        generated_messages.append(
            {"role": "assistant", "content": _decode_answer(tokenizer, content_ids)}
        )

    previous_formatted = ""
    input_ids: list[int] = []
    eligible: list[bool] = []
    assistant_mask: list[bool] = []
    events: list[TraceEvent] = []
    answer_index = 0
    for message_index, message in enumerate(generated_messages):
        is_user = message["role"] == "user"
        formatted = tokenizer.apply_chat_template(
            generated_messages[: message_index + 1],
            tokenize=False,
            add_generation_prompt=is_user,
            add_special_tokens=False,
        )
        chunk_start = _first_differing_index(previous_formatted, formatted)
        chunk_ids = _token_ids(tokenizer, formatted[chunk_start:])
        token_start = len(input_ids)
        input_ids.extend(chunk_ids)
        eligible.extend([True] * len(chunk_ids))
        assistant_mask.extend([False] * len(chunk_ids))
        token_end = len(input_ids)
        if not is_user:
            answer_ids = generated_answers[answer_index]
            answer_index += 1
            offset = next(
                (
                    index
                    for index in range(len(chunk_ids) - len(answer_ids) + 1)
                    if tuple(chunk_ids[index : index + len(answer_ids)]) == answer_ids
                ),
                None,
            )
            if offset is None:
                raise ValueError("Generated assistant content is not present in chat-template chunk")
            response_start = token_start + offset
            valid_answer_end = response_start + len(answer_ids)
            for index in range(response_start, valid_answer_end):
                assistant_mask[index] = True
            for index in range(valid_answer_end, token_end):
                eligible[index] = False
            events.append(
                TraceEvent(
                    event_index=len(events),
                    message_index=message_index,
                    response_start=response_start,
                    valid_answer_end=valid_answer_end,
                    response_end=token_end,
                    generated_token_ids=answer_ids,
                )
            )
        previous_formatted = formatted

    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Generated row has {len(input_ids)} tokens, exceeding "
            f"max_sequence_length={max_sequence_length}; truncation is forbidden"
        )
    record = TraceRecord(
        dataset_index=int(dataset_index),
        puzzle_name=(str(row["puzzle_name"]) if row.get("puzzle_name") is not None else None),
        input_ids=tuple(input_ids),
        eligible_mask=tuple(eligible),
        assistant_target_mask=tuple(assistant_mask),
        events=tuple(events),
    )
    record.validate(max_sequence_length=max_sequence_length)
    return record


def build_final_only_gold_prefix_trace(
    row: Mapping[str, Any],
    *,
    dataset_index: int,
    model: Any,
    tokenizer: Any,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    max_sequence_length: int,
    device: torch.device,
) -> TraceRecord:
    source_messages = row["messages"]
    if not source_messages or len(source_messages) % 2:
        raise ValueError("NVARC row must contain complete user/assistant pairs")
    for message_index in range(0, len(source_messages), 2):
        user = source_messages[message_index]
        assistant = source_messages[message_index + 1]
        if (
            str(user["role"]).lower() != "user"
            or str(assistant["role"]).lower() != "assistant"
        ):
            raise ValueError("NVARC row does not alternate user/assistant")

    # Prior assistant turns are gold demonstrations from the public challenge.
    # Only the final assistant slot is replaced by the model's own generation.
    generated_messages = [
        {"role": str(message["role"]).lower(), "content": message["content"]}
        for message in source_messages[:-1]
    ]
    generated = constrained_greedy_generate(
        model,
        tokenizer,
        generated_messages,
        token_contract=token_contract,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    answer_ids = tuple(generated[:-1])
    if not answer_ids:
        raise RuntimeError("Generated grid has no answer token before EOS")
    generated_messages.append(
        {"role": "assistant", "content": _decode_answer(tokenizer, answer_ids)}
    )

    previous_formatted = ""
    input_ids: list[int] = []
    eligible: list[bool] = []
    assistant_mask: list[bool] = []
    event: TraceEvent | None = None
    final_message_index = len(generated_messages) - 1
    for message_index, message in enumerate(generated_messages):
        is_user = message["role"] == "user"
        formatted = tokenizer.apply_chat_template(
            generated_messages[: message_index + 1],
            tokenize=False,
            add_generation_prompt=is_user,
            add_special_tokens=False,
        )
        chunk_start = _first_differing_index(previous_formatted, formatted)
        chunk_ids = _token_ids(tokenizer, formatted[chunk_start:])
        token_start = len(input_ids)
        input_ids.extend(chunk_ids)
        eligible.extend([True] * len(chunk_ids))
        assistant_mask.extend([False] * len(chunk_ids))
        token_end = len(input_ids)
        if not is_user:
            content_ids = (
                answer_ids
                if message_index == final_message_index
                else tuple(_token_ids(tokenizer, str(message["content"])))
            )
            offset = next(
                (
                    index
                    for index in range(len(chunk_ids) - len(content_ids) + 1)
                    if tuple(chunk_ids[index : index + len(content_ids)])
                    == content_ids
                ),
                None,
            )
            if offset is None:
                raise ValueError(
                    "Assistant content is not present in chat-template chunk"
                )
            response_start = token_start + offset
            valid_answer_end = response_start + len(content_ids)
            for index in range(valid_answer_end, token_end):
                eligible[index] = False
            if message_index == final_message_index:
                for index in range(response_start, valid_answer_end):
                    assistant_mask[index] = True
                event = TraceEvent(
                    event_index=0,
                    message_index=message_index,
                    response_start=response_start,
                    valid_answer_end=valid_answer_end,
                    response_end=token_end,
                    generated_token_ids=answer_ids,
                )
        previous_formatted = formatted

    if event is None:
        raise RuntimeError("Final-only trace did not create its terminal event")
    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Generated row has {len(input_ids)} tokens, exceeding "
            f"max_sequence_length={max_sequence_length}; truncation is forbidden"
        )
    record = TraceRecord(
        dataset_index=int(dataset_index),
        puzzle_name=(
            str(row["puzzle_name"]) if row.get("puzzle_name") is not None else None
        ),
        input_ids=tuple(input_ids),
        eligible_mask=tuple(eligible),
        assistant_target_mask=tuple(assistant_mask),
        events=(event,),
    )
    record.validate(max_sequence_length=max_sequence_length)
    return record


def build_trace_for_row(
    row: Mapping[str, Any],
    **kwargs: Any,
) -> TraceRecord:
    mode = str(row.get("trace_generation_mode", "cumulative_generated"))
    if mode == "cumulative_generated":
        return build_generated_trace(row, **kwargs)
    if mode == "final_only_gold_prefix":
        return build_final_only_gold_prefix_trace(row, **kwargs)
    raise ValueError(f"Unsupported trace_generation_mode: {mode}")


def contiguous_shard_bounds(size: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if size < 0 or num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("Invalid contiguous shard request")
    return size * shard_id // num_shards, size * (shard_id + 1) // num_shards


def completed_indices(shard_dir: str | Path) -> set[int]:
    completed: set[int] = set()
    for path in sorted(Path(shard_dir).glob("chunk-*.safetensors")):
        completion_path = _completion_path(path)
        if not completion_path.is_file():
            continue
        if completion_path.read_bytes() != b"ok\n":
            raise ValueError(f"Trace completion marker is invalid: {completion_path}")
        for record in read_trace_file(path):
            completed.add(record.dataset_index)
    return completed


def write_failures(path: str | Path, failures: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(dict(failure), sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(temporary, destination)
