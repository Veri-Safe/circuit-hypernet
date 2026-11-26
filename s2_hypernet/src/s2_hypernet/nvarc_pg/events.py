from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class PredictionEvent:
    """One current-response target and its prompt-only condition boundary."""

    event_index: int
    message_index: int
    condition_end: int
    response_start: int
    response_end: int
    labels: tuple[int, ...]

    @property
    def response_token_count(self) -> int:
        return self.response_end - self.response_start


@dataclass(frozen=True)
class TokenizedRow:
    """One NVARC row tokenized once with all assistant prediction events."""

    input_ids: tuple[int, ...]
    events: tuple[PredictionEvent, ...]
    dataset_index: int
    puzzle_name: str | None

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_token_count(self) -> int:
        return sum(event.response_token_count for event in self.events)


def _first_differing_index(left: str, right: str) -> int:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    return min(len(left), len(right))


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batched result")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("messages must not be empty")
    normalized = [
        {"role": str(message["role"]).lower(), "content": message["content"]}
        for message in messages
    ]
    for index, message in enumerate(normalized):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(
                "NVARC rows must alternate user/assistant; "
                f"message {index} is {message['role']!r}, expected {expected!r}"
            )
    if normalized[-1]["role"] != "assistant":
        raise ValueError("NVARC training rows must end with an assistant response")
    return normalized


def build_row_events(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    dataset_index: int,
    puzzle_name: str | None,
) -> TokenizedRow:
    """Tokenize one row and expose every assistant position as a prediction event.

    The cumulative-template chunking deliberately mirrors NVARC full-SFT
    tokenization.  For each assistant message, the condition boundary is the
    end of the immediately preceding user chunk, including the generation
    marker but excluding the current gold response.
    """

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    normalized = _normalize_messages(messages)
    previous_formatted = ""
    all_input_ids: list[int] = []
    events: list[PredictionEvent] = []

    for message_index, message in enumerate(normalized):
        role = message["role"]
        formatted = tokenizer.apply_chat_template(
            normalized[: message_index + 1],
            tokenize=False,
            add_generation_prompt=role == "user",
            add_special_tokens=False,
        )
        chunk_start = _first_differing_index(previous_formatted, formatted)
        chunk_ids = _token_ids(tokenizer, formatted[chunk_start:])
        token_start = len(all_input_ids)
        all_input_ids.extend(chunk_ids)
        token_end = len(all_input_ids)

        if role == "assistant":
            if token_end == token_start:
                raise ValueError(f"Assistant message {message_index} produced no tokens")
            labels = [IGNORE_INDEX] * token_end
            labels[token_start:token_end] = all_input_ids[token_start:token_end]
            events.append(
                PredictionEvent(
                    event_index=len(events),
                    message_index=message_index,
                    condition_end=token_start,
                    response_start=token_start,
                    response_end=token_end,
                    labels=tuple(labels),
                )
            )
        previous_formatted = formatted

    if not all_input_ids:
        raise ValueError("Conversation tokenized to zero tokens")
    if len(all_input_ids) > max_sequence_length:
        raise ValueError(
            f"Conversation has {len(all_input_ids)} tokens, exceeding "
            f"max_sequence_length={max_sequence_length}; truncation is forbidden"
        )
    return TokenizedRow(
        input_ids=tuple(all_input_ids),
        events=tuple(events),
        dataset_index=int(dataset_index),
        puzzle_name=puzzle_name,
    )
