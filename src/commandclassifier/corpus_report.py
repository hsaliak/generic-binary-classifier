"""Report corpus coverage and reject development/evaluation command overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from commandclassifier.validate_data import read_jsonl, summarize


def sha256(path: Path) -> str:
    """Return the content hash used to bind a report to exact input files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(development: Path, evaluation: Path) -> dict[str, object]:
    """Summarize validated corpora and their normalized-text overlap."""
    dev = read_jsonl(development)
    ev = read_jsonl(evaluation)
    overlap = sorted({r["text"] for r in dev} & {r["text"] for r in ev})
    return {
        "development": summarize(dev).__dict__,
        "evaluation": summarize(ev).__dict__,
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
    p.add_argument("--output", type=Path)
    a = p.parse_args()
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
