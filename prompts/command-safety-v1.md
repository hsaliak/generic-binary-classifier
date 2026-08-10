# Command Safety Synthetic Dataset Prompt v1

Generate a JSON Lines dataset for a binary classifier of Linux and macOS shell commands. Output only valid JSONL: one JSON object per line, no Markdown, explanation, or code fences.

Generate exactly 50 unique records. Balance `safe` and `unsafe` labels as closely as possible. Vary Bash and Zsh syntax, quoting, paths, arguments, pipelines, redirects, command substitutions, macOS commands, and Linux commands. Do not produce Windows commands.

Use this exact schema for every record:

```json
{"id":"synthetic-v1-001","text":"pwd","label":"safe","family":"read_only_navigation","platform":["linux","macos"],"shell":"bash","risk_reasons":[],"source":"synthetic","generator":"std_slop","prompt_version":"command-safety-v1","batch_id":"replace-with-batch-id","context_required":false}
```

Allowed labels are exactly `safe` and `unsafe`.

Mark `unsafe` when a command can destroy or overwrite data, expose or transmit secrets, execute remotely downloaded content, elevate privilege, weaken security, modify network/firewall/security settings, format or mount disks, kill important processes, or destructively alter source-control history.

Mark `safe` only for low-impact, read-only, or clearly non-destructive commands. Do not label context-dependent destructive commands as safe. Avoid duplicates and near-duplicates.

Use meaningful families such as `read_only_navigation`, `file_deletion`, `permission_change`, `remote_code_execution`, `disk_operation`, `secret_exposure`, `git_history_rewrite`, `process_management`, `package_management`, `network_configuration`, and `shell_obfuscation`.
