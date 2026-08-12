# Prompt Complexity Synthetic Dataset Prompt v1

Generate a JSON Lines dataset for a binary classifier of user prompts written to an LLM assistant. Output only valid JSONL: one JSON object per line, no Markdown, explanation, or code fences.

Generate exactly 50 unique records. Balance `low_complexity` and `high_complexity` labels as closely as possible. Vary prompt length, domain, phrasing style, and punctuation. Prompts are user text only; never include assistant replies.

Use this exact schema for every record:

```json
{"id":"synthetic-v1-001","text":"continue","label":"low_complexity","family":"terse_continuation","complexity_signals":["single atomic continuation"],"source":"synthetic","generator":"std_slop","prompt_version":"prompt-complexity-v1","batch_id":"replace-with-batch-id"}
```

Allowed labels are exactly `low_complexity` and `high_complexity`.

## Complexity rubric

Mark `low_complexity` when the prompt is a single atomic request that an assistant can satisfy in one step:

- terse continuations such as `continue`, `proceed`, `go on`, `again`, `keep going`, `ok`, `next`;
- simple factual recall or short lookup requests;
- one concrete action with no dependencies, constraints, or sub-steps;
- short yes/no or clarification questions.

Mark `high_complexity` when the prompt requires decomposition, sequential multi-step reasoning, coordination across files, systems, or tools, or substantial context synthesis:

- multi-part instructions with ordered or conditional steps;
- tasks spanning several files, modules, repositories, services, or data sources;
- planning, refactoring, or debugging that needs investigation before action;
- synthesizing or comparing large amounts of information into a structured result;
- constraints, formats, and acceptance criteria that must all be satisfied together.

## Boundary rules

`continue`, `proceed`, and similar words alone are always `low_complexity`, even though they reference prior work. A prompt that opens with `continue` but then specifies new multi-step work is `high_complexity`. Produce near-neighbour pairs whose opening token is identical but whose complexity differs, for example `continue` (low) versus `continue and also refactor the module, add tests, and update the docs` (high).

Do not decide on word count alone: a long restatement of one simple fact is low complexity, and a short dense instruction with many constraints can be high complexity. Avoid duplicates and near-duplicates.

Use only these meaningful families:

- `terse_continuation`
- `simple_factual`
- `single_step_action`
- `code_generation`
- `multi_step_planning`
- `debugging_with_context`
- `refactoring`
- `synthesis_research`

Cover every family in every batch. `complexity_signals` must list the concrete reasons for the label from the rubric.