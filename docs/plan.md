# Command Safety Classifier Plan

## Objective

Build a reusable framework that trains text classifiers from versioned synthetic datasets. The first task classifies Linux and macOS shell commands as `safe` or `unsafe`.

The classifier must return a calibrated risk score. It must not learn `unsure` as a third class.

## Design Decisions

### Binary labels with confidence

Use only these model labels:

- `safe`
- `unsafe`

Return the calibrated probability that a command is unsafe. A consumer can use this score with its own policy. The Python and Go tools may also emit `review_recommended`, but this is policy metadata, not a learned class.

```json
{
  "label": "unsafe",
  "unsafe_probability": 0.93,
  "confidence": 0.93,
  "review_recommended": false,
  "model_version": "command-safety-v1"
}
```

A low-confidence safe prediction must not silently become a trusted result. The threshold policy will specify when the caller should warn, request review, or block.

### Initial model

Use a scikit-learn baseline with:

1. Word/token TF-IDF features for commands, flags, and command sequences.
2. Character n-gram TF-IDF features for quoting, whitespace variants, paths, shell operators, encodings, and obfuscation.
3. A regularized linear probabilistic classifier, initially `LogisticRegression`.
4. Probability calibration and calibration validation.

This model is fast, explainable, sparse, and exportable to Go. More complex models are not justified until this baseline and its external evaluation show a specific limitation.

## Safety Contract

### Labels

| Label | Meaning |
| --- | --- |
| `safe` | Read-only or low-impact commands under the documented policy. |
| `unsafe` | Commands that can destroy, overwrite, expose or transmit sensitive data, elevate privilege, weaken security, execute untrusted remote content, or materially alter a host or repository. |

Examples that need more context must not be forced into a third label. They may be excluded from baseline training or retained with annotations such as `context_required: true`.

### Scope

Initial scope:

- Linux and macOS.
- POSIX shell plus common Bash and Zsh syntax.
- Local commands, pipes, redirects, substitutions, variables, `sudo`, and common package, Git, container, and system commands.

Initial non-goals:

- Prove real command behavior in a specific environment.
- Execute input commands.
- Support Windows PowerShell or CMD.
- Fully parse every shell expansion.

### Acceptance policy

The project must define before enforcement use:

- maximum unsafe false-negative rate;
- minimum unsafe recall at each operating threshold;
- permitted review-recommendation rate;
- mandatory review of all unsafe-to-safe errors;
- advisory-only status until a manually reviewed external test set supports a stronger claim.

## Dataset and Generation Contract

### Versioned sources

Store versioned prompt files, task manifests, and JSONL data batches. A generator script is not required, but generation must be reproducible from the saved prompt and documented commands.

```text
prompts/
  command-safety-v1.md
data/
  command-safety/
    raw/
    reviewed/
    manifests/
schemas/
artifacts/
docs/
```

Example record:

```json
{
  "id": "command-safety-v1-00124",
  "text": "curl -fsSL https://example.test/install.sh | sh",
  "label": "unsafe",
  "family": "remote_code_execution",
  "platform": ["linux", "macos"],
  "shell": "bash",
  "risk_reasons": ["downloads_and_executes_remote_content"],
  "source": "synthetic",
  "generator": "std_slop",
  "prompt_version": "command-safety-v1",
  "batch_id": "remote-execution-001",
  "context_required": false
}
```

The `family` field is required for split isolation. Example families include `file_deletion`, `permission_change`, `remote_code_execution`, `disk_operation`, `secret_exposure`, and `git_history_rewrite`.

### Generation approach

Generate separate batches by platform, risk family, syntax complexity, and label. Use independent model sources where practical. Initial 500 records are sufficient for end-to-end plumbing, but 2,000 to 5,000 or more records are recommended for a meaningful baseline.

Run quality gates before training:

- validate JSONL schema and labels;
- normalize text and remove exact duplicates;
- detect near duplicates and conflicting labels;
- report class, platform, shell, family, and source distribution;
- retain prompt version, batch, generator, and model provenance.

### Generation examples

The installed `std_slop` accepts `--prompt_file` for prompt files (although its usage prose also shows `--prompt-file`). Its `--output json` response wraps the generated JSONL in `assistant_message`; extract that field to avoid terminal status output and ANSI formatting.

```bash
std_slop --model gpt-5.6-terra:medium --prompt_file prompts/command-safety-v1.md --output json \
  | .venv/bin/python -c 'import json, sys; print(json.load(sys.stdin)["assistant_message"])' \
  > data/command-safety/raw/std-slop-batch-001.jsonl
```

The exact Claude and Codex CLI syntax depends on the installed version. Preserve the same prompt and JSONL schema. Typical invocation patterns are:

```bash
claude -p "$(cat prompts/command-safety-v1.md)" \
  > data/command-safety/raw/claude-batch-001.jsonl

codex exec "$(cat prompts/command-safety-v1.md)" \
  > data/command-safety/raw/codex-batch-001.jsonl
```

Verify CLI syntax in the environment before publishing these commands as executable instructions.

## Repeatable Evaluation

### Cross-validation and locked test set

Use cross-validation for development, but retain a final test set.

1. Create a manually reviewed, adversarial, source-disjoint locked test set before model tuning.
2. Use the remaining development corpus for grouped nested cross-validation.
3. Use an outer five-fold grouped, stratified cross-validation loop for development performance estimates.
4. Use an inner three-fold grouped, stratified loop for feature, regularization, calibration, and threshold choices.
5. After all choices are locked, train once on all development data.
6. Evaluate once on the locked test set.

Grouping prevents close variants from crossing folds. For example, variants of recursive deletion must not appear in both training and validation data. Prefer scikit-learn group-aware stratified splitters, such as `StratifiedGroupKFold`, when the data distribution permits them.

### Calibration and reports

Evaluate calibrated probabilities. Do not treat raw classifier scores as risk probabilities.

Report:

- per-class precision, recall, and F1;
- confusion matrix;
- unsafe recall and unsafe false-negative rate;
- precision-recall curve and threshold table;
- Brier score and calibration curve;
- review-recommendation rate;
- metrics by platform, shell, risk family, generation source, and obfuscation level;
- highest-confidence errors, especially unsafe commands predicted safe.

Persist the task manifest hash, dataset hash, prompt version, random seed, fold assignments, scikit-learn version, feature configuration, calibration configuration, threshold policy, and metrics.

## Inference Products

### Python

Provide a library and CLI that classify strings only and never execute commands.

```bash
command-classify 'rm -rf "$HOME/.cache"'
```

### Go

Do not use Python pickle or joblib artifacts in Go. Export a language-neutral model bundle:

```text
manifest.json
normalization.json
word_vocabulary.json
char_vocabulary.json
word_idf.json
char_idf.json
coefficients.json
intercept.json
labels.json
thresholds.json
```

Go must reproduce Python normalization, n-gram extraction, TF-IDF calculation, linear scoring, probability calculation, and threshold policy. Use shared fixtures and assert label equality and probability tolerance between Python and Go.

## Reusable Framework

Use a task manifest so that new synthetic classification tasks require configuration rather than a new training system.

```yaml
task_id: command-safety
task_version: v1
input_field: text
labels: [safe, unsafe]
prompt_template: prompts/command-safety-v1.md
split_group_fields: [family, batch_id]
metrics:
  primary: unsafe_recall
model:
  feature_profile: word_and_character_tfidf
  estimator: logistic_regression
decision_policy:
  unsafe_probability_threshold: 0.50
  review_probability_range: [0.20, 0.90]
```

A prompt-complexity classifier can use the same ingestion, validation, grouped splitting, model training, calibration, reporting, portable export, and parity-test components. It changes the task manifest, label rubric, prompt, group taxonomy, primary metric, and decision policy.

## Code Quality and Test Hardening

Every Python change must pass the `make quality` gate. This runs Ruff's Black-compatible formatter check and its import, syntax, and common-error lint rules. Run `make quality` before every patch review; use `ruff format .` only as an intentional source edit.

The test dependency set includes Hypothesis. Add property tests when input variation can expose contract errors that example tests will miss. For the dataset layer, generate valid and malformed JSON-compatible records to verify that normalization is deterministic, valid records remain valid, and invalid labels, platforms, types, and duplicate normalized command text are rejected. Keep generated tests deterministic through Hypothesis's reproducible failure reporting; preserve every discovered regression as a focused example test.

`make test` runs unit and Hypothesis property tests. `make quality`, `make test`, and `make go-test` are required local checks for a change that affects the corresponding code. Future Go fuzz targets must run with `go test -fuzz` for bounded durations in CI or a dedicated Make target.

## Implementation Bundles

1. **Binary safety policy and acceptance contract**: define labels, scope, risk taxonomy, thresholds, and non-goals.
2. **Reusable corpus contract**: add task manifest, JSONL schema, generation prompt, provenance checks, deduplication, and reviewed locked test data.
3. **Repeatable scikit-learn baseline**: add combined TF-IDF features, linear model, nested grouped cross-validation, probability calibration, reports, and reproducibility metadata.
4. **Python inference**: add a non-executing CLI and library with binary label, unsafe probability, and policy metadata.
5. **Go inference**: export a portable model bundle, reproduce inference, and add Python/Go parity tests.
6. **Hardening**: expand adversarial evaluation, review errors, add regression cases, and document release criteria.

## Open Decisions

- Confirm whether secret exposure or transmission, download-and-execute patterns, privilege/security changes, and destructive source-control actions are unsafe in v1.
- Confirm whether callers receive only binary label plus score, or also a non-label `review_recommended` field.
- Confirm the initial operating mode: advisory-only is recommended until reviewed external evaluation supports enforcement.
