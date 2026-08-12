# Framework Details

Operational details behind the [README](../README.md): task definition, data workflow, training, inference, and release. For how to add a new task end-to-end, see the [task creation guide](task-creation/SKILL.md).

## Task definition

A task defines:

- one or more text input fields;
- exactly two labels;
- one positive class;
- grouped train/test split fields;
- decision and review probability policy;
- reviewed development and locked-evaluation datasets.

The training pipeline uses grouped nested evaluation and calibration. It rejects duplicate canonical inputs and development/locked-evaluation overlap. The artifact includes task metadata, input serialization, labels, probability output name, policy, feature metadata, calibration metadata, and input hashes.

Python and Go validate the v3 artifact hash and use the same input serialization. They reject unsupported artifact versions, missing or unknown fields, and empty input values.

## Scope and limits

The current framework supports only binary text classification with one output and one positive class. It does not support multiclass, ordinal, multi-label, multiple-output, arbitrary-estimator, or executable-input tasks.

Generated records are candidate material, not trusted training data. Human review is required before data enters development or locked evaluation.

## Included tasks

`command-safety-v1` classifies Linux and macOS shell-command text as `safe` or `unsafe`; `unsafe` is the positive class. The result is advisory only: a `safe` result is not permission to run a command, and the classifier never runs the command. See the [model card](model-card.md) and [release criteria](release-criteria.md).

`prompt-complexity-v1` classifies a user prompt to an LLM as `low_complexity` or `high_complexity` (`high_complexity` is positive). Terse continuations such as `continue` and `proceed` are low-complexity by design; a prompt that opens with a continuation word but adds new multi-step work is high-complexity. See the [model card](prompt-complexity-model-card.md) and [release criteria](prompt-complexity-release-criteria.md).

Command-safety inference accepts one field:

```bash
commandclassifier \
  --artifact-dir artifacts/command-safety-v1 \
  --text 'rm -rf /tmp/example'
```

The result includes `label`, `positive_probability`, `confidence`, `review_recommended`, `model_version`, and task metadata.

## Create a new task

Start with `templates/binary-task/` and use the working references in `tests/fixtures/generic-multifield/` (multi-field) and `tests/fixtures/generic-positive-first/` (single-field with positive-first label order).

Create a manifest, prompt, schema, raw-data directory, reviewed development dataset, reviewed locked-evaluation dataset, model card, and release criteria. A minimal multi-field manifest is:

```yaml
task_id: request-policy
task_version: v1
input:
  fields: [user_context, candidate_text]
labels:
  values: [allow, block]
  positive_class: block
data:
  prompt: prompts/request-policy-v1.md
  raw_directory: data/request-policy/raw
  processed_dataset: data/request-policy/processed/development.jsonl
  locked_evaluation: data/request-policy/reviewed/locked-eval-v1.jsonl
  split_group_fields: [family, batch_id]
generation:
  backend: std_slop # or claude
```

All configured fields must be present and non-empty. IDs and canonical serialized inputs must be unique. Keep development and locked evaluation disjoint.

Record validation is contract-driven: `validate_data` loads the task manifest and its `record_schema` JSON Schema file, and every runtime check is derived from that contract (labels from the manifest, required/typed/enumerated fields from the schema).

Generation's `std_slop` backend accepts a `--model` override (Makefile: `MODEL=...`) that takes precedence over `generation.model` in the manifest.

## Generate and review candidate material

Generate focused raw candidate material with either configured backend:

```bash
make generate \
  TASK=tasks/request-policy-v1.yaml \
  BACKEND=std_slop \
  BATCH=001 \
  FOCUS='positive boundary cases'
```

Or:

```bash
make generate \
  TASK=tasks/request-policy-v1.yaml \
  BACKEND=claude \
  BATCH=001 \
  FOCUS='positive boundary cases'
```

The Claude backend uses non-interactive `claude -p`. Both backends preserve raw output. Review generated records before merging them into a development corpus. Preserve prompts, backend/model details, batch IDs, review decisions, and hashes.

Create a separate reviewed locked-evaluation dataset and a sibling manifest such as:

```json
{
  "dataset": "locked-eval-v1.jsonl",
  "dataset_sha256": "<sha256>",
  "review_status": "reviewed",
  "reviewer": "reviewer-id",
  "review_date": "YYYY-MM-DD"
}
```

Do not select features, calibrators, thresholds, or policy from locked evaluation data.

## Train a new task

```bash
make train \
  MANIFEST=tasks/request-policy-v1.yaml \
  DATASET=data/request-policy/processed/development.jsonl \
  EVALUATION=data/request-policy/reviewed/locked-eval-v1.jsonl \
  ARTIFACT_DIR=artifacts/request-policy-v1
```

The resulting artifact is `artifacts/request-policy-v1/model.json` with its required `model.json.sha256` sidecar. Python also writes `model.joblib`; Go does not require it.

## Infer on a new task

Use `--text` only for a one-field artifact:

```bash
commandclassifier \
  --artifact-dir artifacts/request-policy-v1 \
  --text 'candidate text'
```

Use `--input-json` for multi-field artifacts:

```bash
commandclassifier \
  --artifact-dir artifacts/request-policy-v1 \
  --input-json '{
    "user_context": "Repository context",
    "candidate_text": "Requested action"
  }'
```

Go uses the same v3 artifact and structured input contract:

```bash
cd go
go run ./cmd/command-classify \
  --model ../artifacts/request-policy-v1/model.json \
  --input-json '{
    "user_context": "Repository context",
    "candidate_text": "Requested action"
  }'
```

For multiple fields, both runtimes serialize fields in manifest order with stable tags. For example:

```text
<USER_CONTEXT>
Repository context

<CANDIDATE_TEXT>
Requested action
```

## Release checklist

Before using a task outside development:

1. Review all training and locked-evaluation records.
2. Verify zero canonical-input overlap.
3. Inspect grouped nested evaluation, calibration, threshold, and locked-evaluation metrics.
4. Verify Python/Go parity for one-field or multi-field task inputs.
5. Write a task-specific model card and release criteria.
6. Keep a human override and review path for high-impact decisions.

## Useful commands

```bash
make help
make quality
make test
make train
make go-test
make go-install
```