# Prompt Complexity Model Card

## Intended use

`prompt-complexity-v1` classifies a user prompt written to an LLM as `low_complexity` or `high_complexity` (`high_complexity` is the positive class). It is advisory-only. It never executes the supplied prompt.

## Output

The artifact returns a high-complexity probability, binary policy label, confidence, review recommendation, and model version. Consumers must treat the probability as an estimate, not a guarantee about the work an assistant will actually perform.

## Complexity rubric

A prompt is `high_complexity` when it requires decomposition, sequential multi-step reasoning, coordination across files, systems, or tools, or substantial context synthesis. A prompt is `low_complexity` when it is a single atomic request, a terse continuation (`continue`, `proceed`, `again`, `ok`, `keep going`, `next`), or simple recall. A prompt that opens with a continuation word but then specifies new multi-step work is `high_complexity`.

## Data and evaluation

Training data is synthetic. The locked reviewed evaluation set has 129 records and is also synthetic. Development has 540 records across 8 families (`terse_continuation`, `simple_factual`, `single_step_action`, `code_generation`, `multi_step_planning`, `debugging_with_context`, `refactoring`, `synthesis_research`). There is zero exact normalized-text overlap between development and locked evaluation. Generated with `std_slop` using `deepseek/deepseek-v4-flash-0731:high`.

## Evaluation results

- Locked-evaluation accuracy: 97.0%
- `high_complexity` recall: 98.6%, precision: 96.1%
- `low_complexity` recall: 94.5%, precision: 98.1%
- Brier score: 0.018 (sigmoid calibration)
- Family breakdown: strong recall on all high-complexity families; `code_generation` recall 88.9% is the lowest (20 support records)

## Limitations

- Synthetic source only; prompts may not match real production traffic in style or domain.
- Complexity is a judgment, not an objective property; different raters may disagree at boundaries.
- Stop-word prompts (`continue`, `proceed`) are classified as low complexity by design, but a prompt that reuses those words with added steps flips to high complexity.
- Go inference parity is covered by the positive-first regression fixture; the same artifact is used by both runtimes.

## Operating policy

Keep v1 advisory-only. Require human review for routing or scheduling decisions based on complexity, and use the artifact's threshold policy rather than hard-coded consumer thresholds. See [release criteria](prompt-complexity-release-criteria.md) and [external evaluation evidence](external-evaluation-evidence.md).