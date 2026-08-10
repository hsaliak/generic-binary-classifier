# Command Safety Model Card

## Intended use

`command-safety-v1` classifies Linux and macOS shell-command text as `safe` or `unsafe`. It is advisory-only. It never executes the supplied text.

## Output

The artifact returns an unsafe probability, binary policy label, confidence, review recommendation, and model version. Consumers must treat the probability as an estimate, not a safety guarantee.

## Data and evaluation

Training data is synthetic. The locked reviewed evaluation set has 39 source-disjoint records and is also synthetic. It has zero exact normalized-text overlap with development data. These facts limit generalization evidence.

## Limitations

- The model can miss harmful commands, including obfuscated or context-dependent commands.
- A `safe` label does not authorize execution.
- It does not analyze command arguments against a live system, identity, filesystem, network, or policy context.
- Go inference is unavailable for production use until parity checks pass.

## Operating policy

Keep v1 advisory-only. Require human review for policy-sensitive decisions and use the artifact's threshold policy rather than hard-coded consumer thresholds. See [external evaluation evidence](external-evaluation-evidence.md) and [release criteria](release-criteria.md).
