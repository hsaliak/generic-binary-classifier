# External Evaluation Evidence

## Purpose

External evidence is required before this classifier can move beyond advisory-only use. It measures performance on command text that was not produced by this project, its prompts, or its model-generation workflow.

## Required inputs

Each release candidate needs a locked evaluation corpus with:

- commands from independently sourced, lawfully usable repositories, manuals, fixtures, or controlled human-authored scenarios;
- documented source URI or internal provenance reference, acquisition date, license/permission status, and content hash;
- two independent human label reviews against the v1 unsafe policy, with recorded adjudication for disagreements;
- Linux and macOS coverage, shell coverage, and coverage for every unsafe policy family;
- realistic safe near-neighbours for destructive, privilege, secret, remote-execution, disk, and source-control commands;
- no normalized-text overlap with development, calibration, training, or prior tuning corpora;
- no use in threshold choice, feature choice, or calibration selection after it is locked.

Do not include real secrets, credentials, personal data, or commands that could identify a private system. Store only minimised command text and the provenance metadata required for audit.

## Maintenance process

1. **Ingest:** preserve the original source reference and record an immutable SHA-256 hash.
2. **Sanitize:** remove secrets and personal data. Record each removal without retaining the removed value.
3. **Label:** two qualified reviewers label independently. An adjudicator resolves differences and records the decision and policy rationale.
4. **Validate:** run schema validation, duplicate/conflict validation, and corpus-overlap checks against every development and tuning corpus.
5. **Lock:** write a reviewed manifest with record count, source provenance, reviewer identities or role IDs, review date, dataset hash, and `evaluation only` usage.
6. **Retain:** keep immutable raw provenance references and reviewed manifests for the supported model lifetime plus the organisation's audit retention period. Restrict access to raw material.
7. **Monitor:** add newly observed false negatives and disputed cases to a quarantine set. Review them periodically; do not add them to a locked benchmark.
8. **Version:** publish a new evaluation version for any content or label change. Never rewrite a locked version.

## Release evidence

A release report must state corpus size, class/family/platform support, unsafe recall, unsafe false-negative rate, Brier score, confidence intervals, and known gaps. Synthetic-only results, including the current 39-record reviewed set, cannot authorize enforcement.
