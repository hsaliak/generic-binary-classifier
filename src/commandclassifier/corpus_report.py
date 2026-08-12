"""Report corpus coverage and reject development/evaluation command overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from commandclassifier.task_definition import TaskDefinition, load_task_definition
from commandclassifier.task_records import read_task_jsonl
from commandclassifier.validate_data import (
    RecordContract,
    load_record_contract,
    read_jsonl,
    summarize,
)


def sha256(path: Path) -> str:
    """Return the content hash used to bind a report to exact input files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    development: Path,
    evaluation: Path,
    task: TaskDefinition | None = None,
    contract: RecordContract | None = None,
) -> dict[str, object]:
    """Summarize corpora and reject canonical input overlap."""
    if task is None and contract is None:
        raise ValueError("build_report requires a task or a record contract")
    if task is None:
        dev = read_jsonl(development, contract)
        ev = read_jsonl(evaluation, contract)
        development_summary = summarize(dev, contract).__dict__
        evaluation_summary = summarize(ev, contract).__dict__
        overlap = sorted(
            {record["text"] for record in dev} & {record["text"] for record in ev}
        )
    else:
        dev = read_task_jsonl(development, task)
        ev = read_task_jsonl(evaluation, task)
        development_summary = {"records": len(dev)}
        evaluation_summary = {"records": len(ev)}
        overlap = sorted(
            {record["_serialized_input"] for record in dev}
            & {record["_serialized_input"] for record in ev}
        )
    return {
        "development": development_summary,
        "evaluation": evaluation_summary,
        "input_sha256": {
            "development": sha256(development),
            "evaluation": sha256(evaluation),
        },
        "exact_text_overlap": overlap,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.manifest is not None:
        task = load_task_definition(a.manifest)
        contract = load_record_contract(a.manifest)
        report = build_report(a.development, a.evaluation, task, contract)
    else:
        report = build_report(a.development, a.evaluation)
    if a.output:
        a.output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    if report["exact_text_overlap"]:
        raise SystemExit("development/evaluation overlap detected")


if __name__ == "__main__":
    main()
