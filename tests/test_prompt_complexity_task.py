import json
from pathlib import Path

import yaml

SCHEMA_PATH = Path("schemas/prompt-complexity-record-v1.json")
MANIFEST_PATH = Path("tasks/prompt-complexity-v1.yaml")


def test_record_schema_is_valid_json_and_references_manifest_labels():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert schema["type"] == "object"
    assert set(manifest["labels"]["values"]) == set(
        schema["properties"]["label"]["enum"]
    )
    assert manifest["data"]["record_schema"] == str(SCHEMA_PATH)


def test_record_schema_requires_all_manifest_metadata_fields():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert set(schema["required"]) == {
        "id",
        "text",
        "label",
        "family",
        "complexity_signals",
        "source",
        "generator",
        "prompt_version",
        "batch_id",
    }
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["complexity_signals"]["type"] == "array"
