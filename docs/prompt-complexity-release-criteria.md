# Prompt Complexity Release Criteria

## Advisory-only artifact release

All conditions must pass:

1. Schema validation (`schemas/prompt-complexity-record-v1.json`), reviewed-manifest validation, and zero development/locked-evaluation overlap.
2. Reproducible training with saved corpus report, metrics, hashes, and artifact contract validation.
3. Python quality and test suite pass.
4. Go parity fixtures, including the positive-first label-order regression test, pass before Go inference is shipped.
5. Model card and external-evidence documentation match the artifact and metrics.

## Routing eligibility

Automated routing, scheduling, or priority decisions based on complexity are prohibited until an independently sourced, human-reviewed locked evaluation set satisfies the process in [external evaluation evidence](external-evaluation-evidence.md). The owner must define and approve minimum high-complexity recall, maximum high-complexity false-negative rate, minimum support for each family, confidence-interval method, and a rollback process before evaluation.

A result from synthetic data alone cannot meet this criterion. Terse-continuation prompts are a deliberate low-complexity class; any routing policy must define how follow-up prompts that add steps are handled.