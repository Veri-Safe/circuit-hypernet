from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from nvarc_qwen3.arc_eval_protocol import (
    ViewSpec,
    build_author_view_specs,
    canonical_view_spec,
    transform_test_view,
    truncate_prompt_to_tokens,
)


@dataclass(frozen=True)
class QueryView:
    query_index: int
    global_test_index: int
    task_id: str
    test_index: int
    is_canonical: bool
    view_index: int
    view_descriptor: str
    prompt: str
    prompt_tokens: int
    removed_train_examples: int
    spec: ViewSpec

    def manifest_record(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "global_test_index": self.global_test_index,
            "task_id": self.task_id,
            "test_index": self.test_index,
            "is_canonical": self.is_canonical,
            "view_index": self.view_index,
            "view_descriptor": self.view_descriptor,
            "prompt_tokens": self.prompt_tokens,
            "removed_train_examples": self.removed_train_examples,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        }


def build_query_views(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[QueryView, ...]:
    max_input_tokens = int(config["prompt"]["max_sequence_length"]) - int(
        config["prompt"]["max_new_tokens"]
    )
    if max_input_tokens <= 0:
        raise ValueError("Frozen evaluator leaves no input-token budget")
    expected_selection_views = int(config["selection_views"]["count"])
    query_views: list[QueryView] = []
    global_test_index = 0

    for task_id in sorted(challenges):
        task = challenges[task_id]
        for test_index in range(len(task["test"])):
            canonical = canonical_view_spec(len(task["train"]))
            selection = build_author_view_specs(
                len(task["train"]),
                augment_n=int(config["selection_views"]["author_augment_n"]),
                seed=int(config["selection_views"]["seed"]),
            )
            if len(selection) != expected_selection_views:
                raise ValueError(
                    f"Expected {expected_selection_views} selection views, found {len(selection)}"
                )
            for position, spec in enumerate((canonical, *selection)):
                transformed = transform_test_view(task, test_index, spec)
                prompt, prompt_tokens, removed = truncate_prompt_to_tokens(
                    transformed,
                    lambda text: tokenizer.encode(text, add_special_tokens=False),
                    max_input_tokens,
                )
                query_views.append(
                    QueryView(
                        query_index=len(query_views),
                        global_test_index=global_test_index,
                        task_id=task_id,
                        test_index=test_index,
                        is_canonical=position == 0,
                        view_index=spec.view_index,
                        view_descriptor=spec.descriptor,
                        prompt=prompt,
                        prompt_tokens=prompt_tokens,
                        removed_train_examples=removed,
                        spec=spec,
                    )
                )
            global_test_index += 1
    return tuple(query_views)


def query_manifest_sha256(query_views: tuple[QueryView, ...]) -> str:
    payload = json.dumps(
        [view.manifest_record() for view in query_views],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
