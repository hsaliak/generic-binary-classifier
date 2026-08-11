# Reviewed Binary Text Classification Framework

This repository is a framework for building **reviewed synthetic binary text-classification tasks**. It trains a calibrated text classifier, exports one portable v3 artifact, and supports equivalent inference in Python and Go.

The framework classifies text only. It never executes supplied text.

## What the framework provides

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

## Included task: dangerous command classification

`command-safety-v1` is the included task. It classifies Linux and macOS shell-command text as `safe` or `unsafe`; `unsafe` is the positive class.

The result is advisory only. A `safe` result is not permission to run a command. The classifier never runs the command. See the [model card](docs/model-card.md), [release criteria](docs/release-criteria.md), and [external evaluation evidence](docs/external-evaluation-evidence.md).

## Quick start: command safety

Requirements: Python 3.11+, Go, and `make`.

```bash
make setup
make quality
make test
make train
make infer TEXT='git status'
make go-infer TEXT='git status'
```

`make train` requires a reviewed, hash-bound locked-evaluation manifest and rejects development/evaluation input overlap before it writes an artifact.

Command-safety inference accepts one field:

```bash
commandclassifier \
  --artifact-dir artifacts/command-safety-v1 \
  --text 'rm -rf /tmp/example'
```

The result includes `label`, `positive_probability`, `confidence`, `review_recommended`, `model_version`, and task metadata.

## Create a new task

Start with `templates/binary-task/` and use the working multi-field reference in `tests/fixtures/generic-multifield/`.

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
  model: configured-model-id
model:
  feature_profile: word_and_character_tfidf
  estimator: logistic_regression
  random_seed: 20250221
evaluation:
  outer_folds: 5
  inner_folds: 3
  primary_metric: positive_recall
decision_policy:
  positive_probability_threshold: 0.50
  review_probability_range: [0.20, 0.90]
```

Quote labels such as `"yes"` and `"no"`; YAML otherwise parses them as booleans.

For multi-field tasks, each record uses an `inputs` object:

```json
{
  "id": "request-001",
  "inputs": {
    "user_context": "Repository contains a production configuration.",
    "candidate_text": "Delete the active configuration."
  },
  "label": "block",
  "family": "configuration_change",
  "batch_id": "batch-001"
}
```

All configured fields must be present and non-empty. IDs and canonical serialized inputs must be unique. Keep development and locked evaluation disjoint.

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

Useful commands:

```bash
make help
make quality
make test
make train
make go-test
make go-install
```

For the detailed task workflow and backend contract, see [the task creation guide](docs/task-creation/SKILL.md).
