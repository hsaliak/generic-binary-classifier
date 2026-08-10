from hypothesis import given
from hypothesis import strategies as st

from commandclassifier.validate_data import normalize_text, validate_record

command_texts = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",), blacklist_characters=("\x00",)
    ),
    min_size=1,
).filter(lambda text: bool(text.strip()))


@given(command_texts)
def test_normalize_text_is_idempotent(text: str):
    assert normalize_text(normalize_text(text)) == normalize_text(text)


@given(
    text=command_texts,
    platform=st.lists(st.sampled_from(["linux", "macos"]), min_size=1, max_size=4),
)
def test_valid_records_remain_valid_after_normalization(text: str, platform: list[str]):
    record = {
        "id": "generated-record",
        "text": text,
        "label": "safe",
        "family": "generated",
        "platform": platform,
        "shell": "bash",
        "risk_reasons": [],
        "source": "test",
        "generator": "hypothesis",
        "prompt_version": "v1",
        "batch_id": "property",
        "context_required": False,
    }

    normalized = validate_record(record)

    assert normalized["text"] == normalize_text(text)
    assert normalized["platform"] == sorted(set(platform))
    assert validate_record(normalized) == normalized
