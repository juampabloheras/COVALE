# COVALE: Compositional Open-Vocabulary Anatomical Localization Evaluation

COVALE evaluates spatial agreement between natural-language anatomical
descriptions by localizing each description within a common anatomical atlas
and measuring overlap between the resulting regions.

```text
reference text -> localization -> atlas ROI --\
                                                -> Dice -> COVALE
candidate text -> localization -> atlas ROI --/
```



## Install and test

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/covale"
uv sync
uv run pytest
```

Copy `.env.example` to `.env` and add a consumer OpenAI API key:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=your-api-key
```

## Calculate COVALE

After populating the atlas registry, calculate a scalar COVALE score with
direct LLM localization. This is the default method:

```python
from covale import covale

score = covale(
    "left frontal lobe",
    "left frontal region",
    registry_path="atlas_registry/registry.json",
)
print(score)
```

Select any method explicitly and optionally override the model:

```python
llm_score = covale(
    "left frontal lobe",
    "left frontal region",
    method="llm",
    model="gpt-6-astra",
)

agent_score = covale(
    "left frontal lobe",
    "left frontal region",
    method="deep_agent",
    model="gpt-6-astra",
)

baseline_score = covale(
    "left frontal lobe",
    "left frontal region",
    method="similarity",
    model="gpt-6-astra",
)
```

The `llm` and `deep_agent` methods independently localize each description to
an atlas expression and calculate Dice overlap. The `similarity` method is a
language-only baseline and does not use the atlas registry.

For a batch, use a JSON file containing an array of phrase pairs:

```json
[
  {
    "reference": "left frontal lobe",
    "candidate": "left frontal region"
  },
  {
    "reference": "left frontal lobe",
    "candidate": "right frontal lobe"
  }
]
```

Load the JSON and evaluate every pair:

```python
import json
from pathlib import Path

from covale import evaluate_batch

pairs = json.loads(Path("pairs.json").read_text(encoding="utf-8"))
results = evaluate_batch(
    pairs,
    registry_path="atlas_registry/registry.json",
    method="llm",
    model="gpt-6-astra",
)

for result in results:
    print(result["reference"], result["candidate"], result["score"])
```

## Run benchmarks

Annotate every line in a JSONL experiment file with its COVALE score and
wall-clock runtime:

```bash
uv run python benchmark/run.py \
  examples/pairs.jsonl \
  benchmark/results/llm.jsonl \
  --method llm
```

The runner supports `llm`, `deep_agent`, and `similarity`, records expected
errors per row without stopping the experiment, and prints aggregate timing
statistics. See [benchmark/README.md](benchmark/README.md) for the output
schema and command options.

## Compare systems

Use the RadEval-style callable and paired bootstrap interface to compare
multiple systems:

```python
from covale import COVALE, compare_systems

covale_evaluator = COVALE(metrics=["dice"], method="llm")
signatures, scores = compare_systems(
    systems={
        "baseline": baseline_descriptions,
        "improved": improved_descriptions,
    },
    metrics={
        "dice": lambda hyps, refs: covale_evaluator(refs, hyps)["dice"],
    },
    references=reference_descriptions,
    n_samples=10_000,
)
```

Each reference/candidate pair is scored once and cached. Bootstrap resampling
uses only those cached scores, so increasing `n_samples` does not make
additional LLM calls. `scores` contains system means and p-values;
`signatures` contains deltas, confidence intervals, significance flags, and
reproducibility settings for each comparison.

## Config file

Use YAML to keep evaluator settings reproducible:

```yaml
metrics:
  - dice:
      method: llm
      provider: openai
      model_name: gpt-6-astra
      registry_path: ../atlas_registry/registry.json

output:
  mode: default

cache: true
```

Load the evaluator from the config:

```python
from covale import COVALE

covale_evaluator = COVALE.from_config("examples/config.yaml")
results = covale_evaluator(refs=refs, hyps=hyps)
```

Output modes are `default`, `per_sample`, and `detailed`. Detailed output
contains the mean Dice score, standard deviation, and per-sample scores. See
[`examples/config.yaml`](examples/config.yaml) for a complete example.


## Adding a new atlas primitive

See [atlas_registry/README.md](atlas_registry/README.md) to add atlas
primitives. 