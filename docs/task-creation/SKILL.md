# Classification Task Creation Skill

Use this skill to add a reviewed synthetic-data **binary text-classification task** to this repository. Command safety is the reference task. Keep the first generic framework scope binary: two labels and one configured positive class.

## Goal

A task definition must provide enough configuration to:

1. generate raw synthetic examples through `std_slop` or `claude -p`;
2. validate, review, and merge data;
3. train a calibrated binary classifier;
4. export one portable artifact;
5. infer through Python and Go with identical input serialization.

Generated data is never automatically trusted. Preserve raw output and require review before training or locking evaluation data.

## Create task files

For task ID `example-task` version `v1`, create:

```text
tasks/example-task-v1.yaml
prompts/example-task-v1.md
schemas/example-task-record-v1.json
data/example-task/raw/
data/example-task/processed/
data/example-task/reviewed/
docs/example-task-model-card.md
```

Start from the command-safety task, but do not copy command-specific labels, fields, or policy names.

## Define the task manifest

The future generic manifest must define:

```yaml
task_id: example-task
task_version: v1

input:
  fields: [text]
  serialization:
    format: single_field

labels:
  values: [allow, block]
  positive_class: block

data:
  prompt: prompts/example-task-v1.md
  raw_directory: data/example-task/raw
  processed_dataset: data/example-task/processed/development.jsonl
  locked_evaluation: data/example-task/reviewed/locked-eval-v1.jsonl
  split_group_fields: [family, batch_id]

generation:
  backend: std_slop # or claude
  model: configured-model-id

model:
  profile: word_and_character_tfidf
  estimator: logistic_regression

evaluation:
  outer_folds: 5
  inner_folds: 3
  primary_metric: positive_recall

decision_policy:
  positive_probability_threshold: 0.50
  review_probability_range: [0.20, 0.90]
```

For multi-field input, specify stable tagged serialization:

```yaml
input:
  fields: [user_context, assistant_text]
  serialization:
    format: tagged_fields
    tags:
      user_context: USER_CONTEXT
      assistant_text: ASSISTANT_TASK
```

Python and Go must serialize these fields identically. Changing serialization requires a new task or artifact version.

## Write the generation prompt

The prompt must define:

- the exact input fields;
- the two allowed labels;
- positive-label policy and boundary cases;
- required families and coverage;
- platform or context coverage, if relevant;
- output JSONL shape;
- prohibition on real secrets, personal data, and unsupported claims;
- distinct examples only.

Generate focused batches, not one broad undirected corpus. Include safe/negative near-neighbours for positive examples.

## Generate raw data

The generic generator interface should accept `TASK`, `BACKEND`, `BATCH`, and `FOCUS`.

Expected commands after generic generation support exists:

```bash
make generate TASK=tasks/example-task-v1.yaml \
  BACKEND=std_slop BATCH=001 FOCUS='positive boundary cases'

make generate TASK=tasks/example-task-v1.yaml \
  BACKEND=claude BATCH=002 FOCUS='negative near-neighbours'
```

`claude` is intentionally a thin backend configuration. Invoke it as `claude -p` with the resolved task prompt and batch instructions. Assume it emits the same extractable record format as `std_slop`; keep its raw output separately and pass it through the same extractor and validator. Do not add backend-specific training logic.

## Validate, review, and lock data

1. Validate each raw extracted batch.
2. Review labels, rationale, inputs, and metadata.
3. Merge reviewed development batches; reject duplicate normalized inputs and conflicting labels.
4. Create a separate reviewed locked evaluation set.
5. Run overlap detection before training.
6. Store reviewed manifests with record count, source provenance, reviewer, review date, and SHA-256 hash.

Never tune against a locked evaluation set.

## Train and release

The generic trainer must use the manifest-defined input fields, labels, positive class, groups, calibration, and policy. It must export task metadata and hashes with the portable artifact.

Before a task is usable:

- grouped nested evaluation passes;
- calibration and threshold metrics are recorded;
- the locked evaluation has zero overlap;
- Python/Go fixtures agree within tolerance;
- the task has a model card and release criteria.

## First generic-framework scope

Implement only generic binary tasks first. Do not add multiclass, ordinal, multi-label, multiple output heads, or arbitrary models in the same change. Add those as later artifact profiles after the binary task contract is stable.
