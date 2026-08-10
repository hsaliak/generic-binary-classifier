# Command Classifier

A reusable framework for synthetic-data text classifiers. Its first task classifies Linux and macOS shell commands as `safe` or `unsafe` and returns a calibrated unsafe probability.

The tool classifies text only. It never executes a command.

## Quick start

```bash
make setup
make test
make go-test
make help
```

See [the implementation plan](docs/plan.md) for the safety contract, data policy, and evaluation design.
