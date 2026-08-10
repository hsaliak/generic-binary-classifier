"""Merge validated synthetic batches into a deduplicated corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commandclassifier.validate_data import (
    RecordValidationError,
    read_jsonl,
    write_jsonl,
)


def merge_batches(paths: list[Path]) -> list[dict[str, object]]:
    """Validate batches, rebase IDs, and reject duplicate normalized commands."""
    merged: list[dict[str, object]] = []
    texts: dict[str, str] = {}
    for path in paths:
        batch_id = path.stem
        for record in read_jsonl(path):
            prior_label = texts.get(record["text"])
            if prior_label is not None:
                if prior_label != record["label"]:
                    raise RecordValidationError(
                        f"{path}: conflicting labels for normalized text "
                        f"{record['text']!r}"
                    )
                continue
            rebased = dict(record)
            rebased["id"] = f"{batch_id}:{record['id']}"
            rebased["batch_id"] = batch_id
            merged.append(rebased)
            texts[record["text"]] = record["label"]
    if not merged:
        raise RecordValidationError("no input records")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge command-classifier JSONL batches."
    )
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = merge_batches(args.input)
    write_jsonl(records, args.output)
    print(json.dumps({"records": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
