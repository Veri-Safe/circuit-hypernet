from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


JsonRecord = dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[JsonRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if not raw_line.endswith("\n"):
                        break
                    raise


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, record: JsonRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, records: Iterable[JsonRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_example_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    return {str(record["example_id"]) for record in iter_jsonl(path) if "example_id" in record}


def count_jsonl(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
