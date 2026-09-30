# Benchmarking

Annotate every line in a JSONL experiment file with its COVALE score and
wall-clock runtime:

```bash
uv run python benchmark/run.py \
  examples/pairs.jsonl \
  benchmark/results/llm.jsonl \
  --method llm \
  --concurrency 8
```

Add `--extract-findings` when the JSONL rows contain report pairs.

See [`benchmark/README.md`](../benchmark/README.md) for the output format and
complete command options.

[Back to the main README](../README.md#additional-resources)
