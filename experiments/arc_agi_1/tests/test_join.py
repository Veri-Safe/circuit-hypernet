from __future__ import annotations

from pathlib import Path

import torch

from s2_hypernet.nvarc_pg.events import build_row_events
from s3circuit.nvarc_circuit.events import CircuitEvent

from s4_circuit.artifacts import CircuitVectorBank, VectorRecord
from s4_circuit.join import build_join_records, load_join_bundle, write_join_bundle


class TinyTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        add_special_tokens: bool,
    ) -> str:
        assert not tokenize and not add_special_tokens
        value = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return value + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _fixture() -> tuple[list[dict], dict[str, CircuitEvent], CircuitVectorBank]:
    raw_rows = [
        {
            "puzzle_name": "p0",
            "source_name": "seed24",
            "source_index": 17,
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
                {"role": "assistant", "content": "D"},
            ],
        }
    ]
    row = build_row_events(
        raw_rows[0]["messages"],
        TinyTokenizer(),
        max_sequence_length=1024,
        dataset_index=0,
        puzzle_name="p0",
    )
    events = {}
    records = {}
    for pair_index, prediction in enumerate(row.events):
        event_key = f"{pair_index + 1:064x}"
        artifact_id = f"{pair_index + 101:064x}"
        event = CircuitEvent(
            event_key=event_key,
            dataset_index=999,
            pair_index=pair_index,
            message_index=prediction.message_index,
            puzzle_name="p0",
            condition_input_ids=tuple(row.input_ids[: prediction.condition_end]),
            gold_response_ids=tuple(
                row.input_ids[prediction.response_start : prediction.response_end]
            ),
            row_token_count=row.length,
            work_unit=pair_index,
            source_name="seed24",
            source_index=17,
        )
        events[event_key] = event
        records[event_key] = VectorRecord(
            event_key=event_key,
            artifact_id=artifact_id,
            shard_relative_path=f"work_unit_{pair_index:04d}/shard.safetensors",
            row_in_shard=0,
            vector=torch.full((59,), float(pair_index), dtype=torch.float32),
        )
    bank = CircuitVectorBank(records, root_manifest={}, final_root=Path("."))
    return raw_rows, events, bank


def test_join_prefers_source_coordinates_and_requires_exact_tokens(tmp_path: Path) -> None:
    raw_rows, events, bank = _fixture()
    records = build_join_records(
        raw_rows,
        tokenizer=TinyTokenizer(),
        max_sequence_length=1024,
        catalog_events=events,
        vector_bank=bank,
    )
    assert [(record.dataset_index, record.pair_index) for record in records] == [
        (0, 0),
        (0, 1),
    ]
    output = tmp_path / "join.safetensors"
    metadata = write_join_bundle(
        output,
        records,
        artifact_contract_sha256="a" * 64,
        catalog_manifest_sha256="b" * 64,
        final_root_manifest_sha256="c" * 64,
    )
    restored, restored_metadata = load_join_bundle(output)
    assert set(restored) == set(events)
    assert restored_metadata["join_sha256"] == metadata["join_sha256"]


def test_join_rejects_token_drift() -> None:
    raw_rows, events, bank = _fixture()
    first = next(iter(events))
    event = events[first]
    events[first] = CircuitEvent(
        **{
            **event.__dict__,
            "gold_response_ids": event.gold_response_ids + (7,),
        }
    )
    try:
        build_join_records(
            raw_rows,
            tokenizer=TinyTokenizer(),
            max_sequence_length=1024,
            catalog_events=events,
            vector_bank=bank,
        )
    except ValueError as error:
        assert "token identity mismatch" in str(error) or "No circuit event matches" in str(error)
    else:
        raise AssertionError("Token drift must fail the strict circuit join")


def test_join_falls_back_to_unique_content_when_direct_coordinate_is_stale() -> None:
    raw_rows = [
        {
            "puzzle_name": "p0",
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ],
        },
        {
            "puzzle_name": "p1",
            "messages": [
                {"role": "user", "content": "X"},
                {"role": "assistant", "content": "Y"},
            ],
        },
    ]
    tokenized = [
        build_row_events(
            row["messages"],
            TinyTokenizer(),
            max_sequence_length=1024,
            dataset_index=index,
            puzzle_name=row["puzzle_name"],
        )
        for index, row in enumerate(raw_rows)
    ]
    correct = tokenized[0].events[0]
    stale = tokenized[1].events[0]
    correct_key = "1" * 64
    stale_key = "2" * 64
    events = {
        correct_key: CircuitEvent(
            event_key=correct_key,
            dataset_index=99,
            pair_index=0,
            message_index=correct.message_index,
            puzzle_name="p0",
            condition_input_ids=tuple(tokenized[0].input_ids[: correct.condition_end]),
            gold_response_ids=tuple(
                tokenized[0].input_ids[correct.response_start : correct.response_end]
            ),
            row_token_count=tokenized[0].length,
            work_unit=0,
        ),
        stale_key: CircuitEvent(
            event_key=stale_key,
            dataset_index=0,
            pair_index=0,
            message_index=stale.message_index,
            puzzle_name="p1",
            condition_input_ids=tuple(tokenized[1].input_ids[: stale.condition_end]),
            gold_response_ids=tuple(
                tokenized[1].input_ids[stale.response_start : stale.response_end]
            ),
            row_token_count=tokenized[1].length,
            work_unit=0,
        ),
    }
    bank = CircuitVectorBank(
        {
            key: VectorRecord(
                event_key=key,
                artifact_id=f"{index + 1:064x}",
                shard_relative_path=f"work_unit_{index:04d}/shard.safetensors",
                row_in_shard=0,
                vector=torch.zeros(59),
            )
            for index, key in enumerate(events)
        },
        root_manifest={},
        final_root=Path("."),
    )

    records = build_join_records(
        raw_rows,
        tokenizer=TinyTokenizer(),
        max_sequence_length=1024,
        catalog_events=events,
        vector_bank=bank,
    )

    assert [record.event_key for record in records] == [correct_key, stale_key]
