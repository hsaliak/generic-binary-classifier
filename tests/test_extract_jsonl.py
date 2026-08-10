import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from commandclassifier.extract_jsonl import main


def test_extract_jsonl_keeps_records_and_drops_prose():
    response = {"assistant_message": '{"id":"one"}\n### STATE\n{"id":"two"}'}
    output = io.StringIO()
    with patch("sys.stdin", io.StringIO(json.dumps(response))), redirect_stdout(output):
        main()

    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"id": "one"},
        {"id": "two"},
    ]
