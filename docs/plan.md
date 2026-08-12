# Generic Binary Classification Framework Plan

## Current objective

Convert this repository from a command-safety-specific classifier into a reusable framework for **reviewed synthetic binary text-classification tasks**.

`command-safety-v1` is the first configured task. It must continue to train and infer while the framework is generalized.

The first generic release supports:

- one task-defined binary label pair;
- one configured positive class;
- one calibrated binary classifier;
- one portable artifact;
- Python and Go inference;
- one or more task-defined input fields.

It does not yet support multiclass, ordinal, multi-label, multiple-output, or arbitrary-estimator tasks.

## Operating requirements

- Classifiers classify text only; they never execute input text.
- Synthetic data is raw candidate material, never automatically trusted training data.
- Preserve raw generator output, prompt version, backend, model ID, batch ID, review decisions, and hashes.
- Require human review before data enters development or locked evaluation sets.
- Reject normalized development/evaluation input overlap before training.
- Keep locked evaluation data out of model, feature, calibration, and threshold selection.
- Keep command safety advisory-only until independently sourced evaluation satisfies its release criteria.

## Target task definition

A generic binary task defines:

```yaml
task_id: example-task
task_version: v1
input:
  fields: [text]
labels:
  values: [allow, block]
  positive_class: block
data:
  split_group_fields: [family, batch_id]
generation:
  backend: std_slop # or claude
model:
  profile: word_and_character_tfidf
  estimator: logistic_regression
decision_policy:
  positive_probability_threshold: 0.50
  review_probability_range: [0.20, 0.90]
```

For multiple inputs, the artifact will define a stable tagged serialization, for example:

```text
<USER_CONTEXT>
...

<ASSISTANT_TASK>
...
```

Python and Go must serialize fields identically.

## Implementation progress

### Bundle 1 — Generic task definition and binary contract `[x]`

Completed:

- Added `src/generic_binary_classifier/task_definition.py`.
- Added validated task properties:
  - task ID and version;
  - input fields;
  - exactly two unique labels;
  - positive class;
  - split groups;
  - positive decision threshold;
  - review probability range.
- Added deterministic single-field and multi-field tagged input serialization.
- Migrated `tasks/command-safety-v1.yaml` to generic `input`, `labels`, `data`, `generation`, `model`, `evaluation`, and `decision_policy` sections.
- Updated training to obtain groups and policy from the task definition.
- Added task-definition tests.

Important rule: quote YAML labels such as `yes` and `no`; YAML otherwise interprets them as booleans.

### Bundle 2 — Generic records and input serialization `[x]`

Completed:

- Added `src/generic_binary_classifier/task_records.py`.
- Supports existing command-safety records with top-level `text`.
- Supports generic records with an explicit `inputs` object.
- Validates task-defined labels and input fields.
- Normalizes and serializes inputs into `_serialized_input`.
- Rejects duplicate IDs and duplicate normalized serialized inputs.
- Training and locked-evaluation prediction now use `_serialized_input`.
- Added multi-field serialization and duplicate-input tests.

Example future generic record:

```json
{
  "id": "example-001",
  "inputs": {
    "user_context": "Repository context",
    "assistant_text": "Requested work"
  },
  "label": "block",
  "family": "example_family",
  "batch_id": "001"
}
```

### Bundle 3 — Generic trainer and portable artifact v3 `[x]`

Completed.

Required work:

- Remove fixed `safe`/`unsafe` assumptions from training metrics, positive-target construction, threshold tables, and evaluation labels.
- Use the configured task labels and positive class throughout grouped nested training and calibration.
- Export portable artifact format v3 with:
  - task ID/version;
  - input fields and serialization metadata;
  - labels and positive class;
  - task-defined probability output field;
  - threshold/review policy;
  - calibration and feature metadata;
  - dataset and task-manifest hashes.
- Regenerate command-safety artifact through v3.
- Preserve strict artifact-version rejection; do not reinterpret v2 artifacts as v3.

### Bundle 4 — Generic Python and Go inference `[x]`

Required work:

- Add `--input-json` for structured multi-field input.
- Retain `--text` only for one-field artifacts.
- Validate missing and unknown fields.
- Serialize inputs from artifact metadata.
- Emit configured positive probability and task metadata instead of fixed `unsafe_probability` fields.
- Update Go artifact parsing, structured input parsing, output construction, and contract validation for v3.
- Add shared Python/Go fixtures for one-field and multi-field tasks.

### Bundle 5 — Thin configured generation and templates `[x]`

Required work:

- Make `TASK`, `BACKEND`, `BATCH`, and `FOCUS` task-aware generation inputs.
- Keep `std_slop` as the current backend.
- Add a thin `claude -p` backend configuration.
  - It receives the resolved task prompt and batch instructions.
  - It is assumed to emit the same extractable record shape.
  - It uses the same raw-output, extraction, validation, review, and merge flow.
  - Do not add backend-specific training logic.
- Add a binary task template with manifest, prompt, schema, data directories, and model-card skeleton.

### Bundle 6 — Generic regression and CI gates `[x]`

Required work:

- Run command safety through the generic v3 pipeline.
- Add a small second binary fixture task with multi-field input.
- Add generic manifest, serialization, training, artifact, Python, and Go parity tests.
- Add bounded Go fuzzing for structured input and v3 artifact parsing.
- Update CI to run both command-safety regression and generic fixture-task checks.

## Current validation checkpoint

The current checkpoint passed:

```bash
make quality
make test              # 33 passed
make train
make go-infer TEXT='git status'
```

Python and Go command-safety inference still agree within floating-point noise for `git status`.

## Known current limitation

Artifacts and inferencers use the generic v3 contract. The framework remains intentionally limited to reviewed binary text-classification tasks with one configured positive class.

## References

- [Task creation skill](task-creation/SKILL.md)
- [Command-safety model card](model-card.md)
- [External evaluation evidence](external-evaluation-evidence.md)
- [Release criteria](release-criteria.md)
