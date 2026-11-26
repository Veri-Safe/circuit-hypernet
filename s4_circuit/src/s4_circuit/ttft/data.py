from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s2_hypernet.nvarc_pg.events import TokenizedRow, build_row_events
from s4_circuit.contracts import semantic_sha256


LOADER_SHA256 = "d01cd56167e534ae156706ef62ab11662fc7b403964614a7a55131940bde970c"
DATA_PROTOCOL = "nvarc_author_ttft_aug128_all_assistant_v1"


def author_loader(root: Path) -> dict[str, Any]:
    """Load only the pinned, CPU-only arc_loader cell, never the notebook runner."""
    path = root / "ref_repo/NVARC/ARC-AGI1/002_ivan_arc1.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    marker = "%%writefile arc_loader.py\n"
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    sources = [source[len(marker):] for source in sources if source.startswith(marker)]
    if len(sources) != 1:
        raise ValueError("Expected exactly one NVARC arc_loader cell")
    source = sources[0]
    if hashlib.sha256(source.encode()).hexdigest() != LOADER_SHA256:
        raise ValueError("NVARC augmentation source changed; review before running TTFT")
    tree = ast.parse(source)
    # AutoTokenizer is used only as an annotation in this cell.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "transformers"
    )]
    namespace: dict[str, Any] = {"AutoTokenizer": Any}
    exec(compile(tree, str(path) + ":arc_loader", "exec"), namespace)
    return namespace


@dataclass(frozen=True)
class SupportRow:
    task_id: str
    view_index: int
    descriptor: str
    removed_examples: int
    messages: tuple[dict[str, str], ...]

    def tokenize(self, tokenizer: Any) -> TokenizedRow:
        return build_row_events(
            self.messages, tokenizer, max_sequence_length=8192,
            dataset_index=self.view_index, puzzle_name=self.task_id,
        )


def build_support_rows(task_id: str, task: dict, tokenizer: Any, root: Path) -> tuple[SupportRow, ...]:
    author = author_loader(root)
    formatter = author["QwenFormatter"](tokenizer)
    # Deliberately exclude real test inputs and never load solutions here.
    example = task["train"][0]["input"]
    support_only = {task_id: {"train": task["train"], "test": [{"input": example}]}}
    state = np.random.get_state()
    try:
        dataset = author["ArcDataset"](support_only).augment(n=16, shfl_keys=True, seed=1)
        dataset = dataset.cut_to_len(formatter=formatter, name="text", max_len=8192)
    finally:
        np.random.set_state(state)
    rows = []
    for index, key in enumerate(dataset.keys):
        pairs = dataset.queries[key]["train"]
        if not pairs:
            raise ValueError(f"TTFT cannot silently train an empty puzzle: {task_id}")
        messages = tuple(message for pair in pairs for message in (
            {"role": "user", "content": author["convert_grid_to_string"](pair["input"])},
            {"role": "assistant", "content": author["convert_grid_to_string"](pair["output"])},
        ))
        row = SupportRow(task_id, index, key, len(task["train"]) - len(pairs), messages)
        tokenized = row.tokenize(tokenizer)
        official_ids = tuple(tokenizer.encode(dataset.get(key, formatter)["text"], add_special_tokens=False))
        if tokenized.input_ids != official_ids:
            raise ValueError(f"Author text/chat-template token parity failed: {task_id}/{index}")
        rows.append(row)
    if len(rows) != 128:
        raise ValueError("Author TTFT must produce 128 rows per puzzle")
    return tuple(rows)


def support_key(task_id: str, condition_ids: tuple[int, ...], identity: dict) -> str:
    # No current gold response, future examples, PG checkpoint or LR in this key.
    return semantic_sha256({
        "format": "s4_ttft_support_condition_v1", "task_id": task_id,
        "condition_ids": condition_ids, "circuit_identity": identity,
    })


def condition_requests(rows: tuple[SupportRow, ...], tokenizer: Any, identity: dict) -> dict[str, dict]:
    requests: dict[str, dict] = {}
    for row in rows:
        tokens = row.tokenize(tokenizer)
        for event in tokens.events:
            condition = tokens.input_ids[:event.condition_end]
            key = support_key(row.task_id, condition, identity)
            requests.setdefault(key, {
                "event_key": key, "task_id": row.task_id,
                "condition_input_ids": list(condition),
                "condition_sha256": semantic_sha256(condition),
                "view_index": row.view_index, "pair_index": event.event_index,
                "identity": identity,
            })
    return requests


def rows_identity(rows: tuple[SupportRow, ...]) -> str:
    return semantic_sha256([asdict(row) for row in rows])
