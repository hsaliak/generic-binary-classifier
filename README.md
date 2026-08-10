# Command Classifier

Command Classifier is a reusable framework for training versioned text classifiers from reviewed datasets. The first task, `command-safety-v1`, classifies Linux and macOS shell-command text as `safe` or `unsafe` and returns a calibrated unsafe probability.

The Python and Go tools classify text only. They never execute the supplied command.

> **Operating status:** v1 is advisory-only. A `safe` result is not permission to execute a command. See the [model card](docs/model-card.md) and [release criteria](docs/release-criteria.md).

## Quick start

Requirements: Python 3.11+, Go 1.25+, and `make`.

```bash
make setup
make quality
make test
make corpus-report
make train
```

`make train` validates the reviewed evaluation manifest and rejects development/evaluation overlap before it writes an artifact.

## Synthetic data workflow

The development corpus is generated in focused batches using `std_slop` and the task prompt in `prompts/command-safety-v1.md`.

```bash
make generate BATCH=013 FOCUS='quoted destructive commands and safe near-neighbours'
```

Generation output is raw input, not training-ready data. Validate, review, and merge batches before training:

```bash
make validate-data DATASET=data/command-safety/raw/std-slop-batch-013.jsonl
python -m commandclassifier.merge_data \
  --input data/command-safety/raw/*.jsonl \
  --output data/command-safety/processed/development.jsonl
make corpus-report
```

Synthetic data is useful for development coverage but cannot authorize enforcement. Preserve prompts, generator details, review decisions, hashes, and manifests. For independently sourced evaluation evidence and maintenance requirements, see [external evaluation evidence](docs/external-evaluation-evidence.md).

## Python inference

Train first if `artifacts/command-safety-v1/` does not contain a current artifact:

```bash
make train
make infer TEXT='git status'
make infer TEXT='rm -rf /tmp/example'
```

The Python CLI verifies the portable bundle hash and Python model hash. It emits JSON with `label`, `unsafe_probability`, `confidence`, `review_recommended`, and `model_version`.

## Go inference

The Go runtime reads the portable `model.json` and its required `model.json.sha256` sidecar. It does not require Python or `model.joblib` at inference time.

For development use:

```bash
make train
make go-infer TEXT='git status'
make go-infer TEXT='git reset --hard HEAD~3'
```

To install the Go binary locally:

```bash
make go-install
./bin/command-classify \
  --model artifacts/command-safety-v1/model.json \
  --text 'git status'
```

To deploy it separately, copy both portable artifact files with the binary:

```text
command-classify
model.json
model.json.sha256
```

The portable JSON contains the model weights: vocabularies, IDF values, linear coefficients, intercept, calibration mapping, normalization metadata, and decision policy. The Python-only `model.joblib` is not required by Go. Keep the bundle and sidecar together; Go fails if the sidecar hash does not match.

Test the Go implementation with:

```bash
make go-test
cd go && go test -run=^$ \
  -fuzz=FuzzClassifyNeverReturnsInvalidProbability \
  -fuzztime=5s ./cmd/command-classify
```

## Creating another task

The data contract is reusable: create a new task manifest, prompt, raw/processed/reviewed dataset directories, and a locked reviewed evaluation manifest. Keep labels mutually exclusive, record provenance and review metadata, and require overlap checks before training.

The current trainer and portable artifact are intentionally specialized to binary `safe`/`unsafe` scoring: they name `unsafe` as the positive class and emit command-safety policy fields. To make a fully generic task:

1. Add task-configured labels, positive class, model version, and decision-policy output fields.
2. Replace the fixed `safe`/`unsafe` checks in `train.py`, `cli.py`, exporter, and Go runtime with validated manifest configuration.
3. Define task-specific prompt, schema extensions, class balance, grouped split fields, and review policy.
4. Add a task-specific locked evaluation corpus and model card.
5. Add Python/Go fixtures for the new artifact configuration before production use.

Do not weaken the review, provenance, overlap, calibration, or release-evidence gates when adding a task.

### Worked example: agent task routing

A useful future task can estimate the resources an agent task is likely to need from the **user context** and **assistant task text**. For example, it can recommend a reasoning tier (`low`, `medium`, or `high`) and a model-capability tier (`weak`, `medium`, or `strong`). These are routing recommendations, not guarantees of task quality or successful completion.

Use two independently reviewed targets rather than combining the two dimensions into nine opaque labels:

```json
{
  "id": "route-0001",
  "user_context": "Repository is a Python command classifier with an existing test suite.",
  "assistant_text": "Add a Make target and update the README with usage examples.",
  "reasoning_effort": "low",
  "model_capability": "weak",
  "rationale": ["localized documentation and build-file change", "no cross-language parity work"],
  "source": "human_authored",
  "review_status": "reviewed"
}
```

A higher-resource example could be:

```json
{
  "id": "route-0002",
  "user_context": "A portable Python/Go model artifact must give equivalent probabilities.",
  "assistant_text": "Implement Unicode-compatible TF-IDF, calibration, shared fixtures, fuzzing, and release gates in Go.",
  "reasoning_effort": "high",
  "model_capability": "strong",
  "rationale": ["cross-language numerical parity", "Unicode and calibration edge cases", "safety release impact"],
  "source": "human_authored",
  "review_status": "reviewed"
}
```

Implement this as two binary or ordinal classifiers after the generic-task refactor:

- `reasoning_effort`: train ordered decisions such as `high_or_lower` and `medium_or_lower`, then derive `low`, `medium`, or `high` from calibrated probabilities; or add a validated multiclass artifact contract.
- `model_capability`: use the same ordered approach for `weak`, `medium`, and `strong`.

Create separate development, calibration, and locked evaluation sets. Split by task family, product area, authoring source, or request template so near-duplicate tasks cannot leak across splits. Review routing outcomes with human operators, measure per-tier confusion and costly under-routing, and keep an explicit manual-override path. Do not use a generated recommendation as the only control for high-impact agent actions.

A task manifest can define the input fields and output policy after the trainer becomes generic:

```yaml
task_id: agent-routing
task_version: v1
input_fields:
  - user_context
  - assistant_text
outputs:
  - name: reasoning_effort
    labels: [low, medium, high]
  - name: model_capability
    labels: [weak, medium, strong]
split_group_fields:
  - task_family
  - source_batch
```

This example cannot run through the current `command-safety-v1` trainer unchanged. That trainer has a fixed binary `safe`/`unsafe` contract. Complete the five generic-task changes above, then add task-specific schema validation, Python/Go fixtures, evaluation criteria, and a model card before using this routing task in production.

## Useful commands

```bash
make help
make quality
make test
make corpus-report
make train
make infer TEXT='git status'
make go-infer TEXT='git status'
make go-install
make go-test
```
