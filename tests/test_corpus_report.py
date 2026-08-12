from pathlib import Path

from generic_binary_classifier.corpus_report import build_report
from generic_binary_classifier.validate_data import load_record_contract, write_jsonl

CONTRACT = load_record_contract(Path("tasks/command-safety-v1.yaml"))


def record(record_id: str, text: str, label: str) -> dict[str, object]:
    return {
        "id": record_id,
        "text": text,
        "label": label,
        "family": "test_family",
        "platform": ["linux"],
        "shell": "bash",
        "risk_reasons": [],
        "source": "test",
        "generator": "test",
        "prompt_version": "v1",
        "batch_id": "batch-1",
        "context_required": False,
    }


def test_build_report_binds_zero_overlap_result_to_input_hashes(tmp_path: Path):
    development = tmp_path / "development.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    write_jsonl([record("development", "pwd", "safe")], development)
    write_jsonl([record("evaluation", "rm -f tmp", "unsafe")], evaluation)

    report = build_report(development, evaluation, contract=CONTRACT)

    assert report["exact_text_overlap"] == []
    assert len(report["input_sha256"]["development"]) == 64
    assert report["development"]["records"] == 1
    assert report["evaluation"]["records"] == 1


def test_build_report_detects_normalized_text_overlap(tmp_path: Path):
    development = tmp_path / "development.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    write_jsonl([record("development", "pwd", "safe")], development)
    write_jsonl([record("evaluation", " pwd ", "safe")], evaluation)

    report = build_report(development, evaluation, contract=CONTRACT)

    assert report["exact_text_overlap"] == ["pwd"]
