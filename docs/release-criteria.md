# Release Criteria

## Advisory-only artifact release

All conditions must pass:

1. Schema validation, reviewed-manifest validation, and zero development/evaluation overlap.
2. Reproducible training with saved corpus report, metrics, hashes, and artifact contract validation.
3. Python quality and test suite pass.
4. Go parity fixtures, probability tolerance tests, malformed-artifact tests, and bounded fuzz tests pass before Go inference is shipped.
5. Model card and external-evidence documentation match the artifact and metrics.

## Enforcement eligibility

Enforcement is prohibited until an independently sourced, human-reviewed locked evaluation set satisfies the process in [external evaluation evidence](external-evaluation-evidence.md). The owner must define and approve minimum unsafe recall, maximum unsafe false-negative rate, minimum support for each policy family/platform, confidence-interval method, and a rollback process before evaluation.

A result from synthetic data alone cannot meet this criterion.
