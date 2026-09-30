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

## COVALE and RadEval metric comparison

`compare_metrics.py` evaluates 30 predefined anatomical phrase pairs using:

- COVALE `llm`
- COVALE `deep_agent`
- COVALE `similarity`
- BLEU
- ROUGE-1, ROUGE-2, and ROUGE-L
- BERTScore
- RadEval-BERTScore

The default benchmark runs every pair 10 times. Each COVALE pair/repetition
constructs a fresh evaluator with `cache=False`, ensuring repeated model calls
are independent. RadEval performs a fresh metric call for every repetition
while retaining loaded model weights.

### Install the additional benchmark dependency

From the repository root:

```bash
uv pip install \
  --python .venv/bin/python \
  -r benchmark/requirements.txt
```

If COVALE is installed in a different environment, replace
`.venv/bin/python` with that environment's Python executable.

Create a repository-root `.env` and add the consumer OpenAI credential:

```dotenv
OPENAI_API_KEY=your-openai-api-key
```

Add `TAVILY_API_KEY` to the same file to enable optional Deep Agent anatomy
research. The benchmark stops before starting if COVALE methods are selected
and `OPENAI_API_KEY` is unavailable.

The BERTScore metrics download model weights from Hugging Face on first use.
Setting `HF_TOKEN` is optional but increases Hub rate limits:

```bash
export HF_TOKEN=your-hugging-face-token
```

### Smoke test

Run every metric on the first phrase pair once:

```bash
.venv/bin/python benchmark/compare_metrics.py \
  --smoke \
  --overwrite \
  --output-dir benchmark/results/smoke
```

### Full benchmark

Run all 30 pairs for 10 repetitions:

```bash
.venv/bin/python benchmark/compare_metrics.py \
  --repetitions 10 \
  --workers 1 \
  --overwrite \
  --output-dir benchmark/results/metric_comparison
```

`--workers` controls independent concurrent COVALE calls. Increase it
cautiously to avoid provider rate limits. The complete default run performs
900 COVALE evaluations plus Deep Agent prechecks and tool calls.

### Quick benchmark

Run five representative pairs for three independent repetitions:

```bash
.venv/bin/python benchmark/compare_metrics.py \
  --quick \
  --workers 2 \
  --overwrite \
  --output-dir benchmark/results/quick
```

The quick set includes semantic equivalence, subregion containment, lobar
containment, and spatially disjoint or adjacent anatomy. It performs 45 COVALE
evaluations and all RadEval metrics.

Interrupted runs can resume by repeating the command without `--overwrite`:

```bash
.venv/bin/python benchmark/compare_metrics.py \
  --repetitions 10 \
  --workers 1 \
  --output-dir benchmark/results/metric_comparison
```

### Outputs

- `raw_results.jsonl`: every pair, metric, repetition, timing, expression,
  diagnostic, and error
- `summary.csv` and `summary.json`: per-pair score variability
- `metric_summary.csv` and `metric_summary.json`: overall metric comparison
  including score variance across repetitions and average per-pair completion
  time
- `manifest.json`: package versions, inputs, methods, repetition count, and
  cache/call settings
