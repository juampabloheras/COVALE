# Benchmark runner

The benchmark runner reads JSONL phrase pairs sequentially and writes an
annotated JSONL file. Each non-empty input line must contain `reference` and
`candidate` strings.

```bash
uv run python benchmark/run.py \
  examples/pairs.jsonl \
  benchmark/results/llm.jsonl \
  --method llm \
  --concurrency 8
```

Available methods are `llm`, `deep_agent`, and `similarity`. Use `--model` to
override `gpt-6-astra`, `--registry` to select an atlas registry,
`--concurrency` for bounded ordered execution, `--summary` for a custom
summary path, and `--overwrite` to replace existing output. Pass
`--extract-findings` when each reference and candidate is a report rather
than an anatomical location.

Successful rows receive:

```json
{
  "covale": {
    "status": "ok",
    "score": 0.8,
    "elapsed_seconds": 1.23,
    "method": "llm",
    "model": "gpt-6-astra",
    "line": 1
  }
}
```

Expected row, localization, and provider failures are written with
`"status": "error"` and processing continues. The command prints aggregate
timing and count statistics, including separate provider setup time, then
exits non-zero if any rows failed. The selected OpenAI client or Deep Agent is
created once and reused for every row.

The default sidecar path is `<output>.summary.json`. It records SHA-256 hashes
for inputs, outputs, registry, and prompt; package/Python versions; Git commit;
atlas metadata; finding-extraction settings; counts; concurrency; and timing.
