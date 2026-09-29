# COVALE: Compositional Open-Vocabulary Anatomical Localization Evaluation

Text-based metrics can tell whether two anatomical descriptions use similar
words, but they often miss whether the descriptions point to the same place.
COVALE turns each description into a 3D region in an anatomical atlas, then
measures how much the regions overlap. This makes it easy to compare generated
descriptions, evaluate systems, and provide spatial feedback during model
training.

```text
reference text -> localization -> atlas ROI --\
                                                -> Dice -> COVALE
candidate text -> localization -> atlas ROI --/
```

## Table of contents

- [Install and test](#install-and-test)
- [Calculate COVALE](#calculate-covale)
- [Run benchmarks](#run-benchmarks)
- [Compare systems](#compare-systems)
- [Config file](#config-file)
- [RL rewards](#rl-rewards)
- [Adding a new atlas primitive](#adding-a-new-atlas-primitive)


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

| Method | Description | Uses atlas masks |
|---|---|---|
| `llm` (default) | Uses OpenAI to map each description to an atlas region, then calculates Dice overlap | Yes |
| `deep_agent` | Uses a Deep Agent to map each description to an atlas region, then calculates Dice overlap | Yes |
| `similarity` | Uses OpenAI to compare the descriptions directly as a language-only baseline | No |

After populating the atlas registry, calculate a scalar COVALE score:

```python
from covale import COVALE

covale_evaluator = COVALE(
    metrics=["dice"],
    method="llm",
    model="gpt-6-astra",
    registry_path="atlas_registry/registry.json",
    provider="openai",
    per_sample=False,
    output_mode=None,
    cache=True,
    concurrency=1,
    errors="raise",
    client=None,
    agent=None,
)
results = covale_evaluator(
    refs=["left frontal lobe"],
    hyps=["left frontal region"],
)
print(results["dice"])
```


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
  --method llm \
  --concurrency 8
```

See [benchmark/README.md](benchmark/README.md) for the output
format and command options.


## Compare systems


```python
from covale import COVALE, compare_systems

covale_evaluator = COVALE()
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
    test="bootstrap",
)
```

`scores` contains system means and p-values;
`signatures` contains deltas, confidence intervals, significance flags, and
reproducibility settings for each comparison.

Set `test="approximate_randomization"` for a paired randomization test instead
of paired bootstrap.

## Config file

Use YAML to keep evaluator settings reproducible:

```yaml
metrics:
  - dice:
      method: llm
      provider: openai
      model_name: gpt-6-astra
      registry_path: ../atlas_registry/registry.json
      concurrency: 4

output:
  mode: detailed
  errors: record
```

Load the evaluator from the config:

```python
from covale import COVALE

covale_evaluator = COVALE.from_config("examples/config.yaml")
results = covale_evaluator(refs=refs, hyps=hyps)
```

| Output mode | Result |
|---|---|
| `default` | Mean Dice score |
| `per_sample` | Dice score for each reference/candidate pair |
| `detailed` | Mean and standard deviation, per-sample scores, localization expressions and timings, expression complexity, resolved fraction, and categorized failures |

The default `errors: raise` stops when a pair cannot be evaluated.
`errors: record` is available with `mode: detailed` and includes unresolved
pairs in the result. See [`examples/config.yaml`](examples/config.yaml) for a
complete example.

## RL rewards

Install the optional TRL integration:

```bash
uv sync --extra rl
```

Create a TRL-compatible reward function:

```python
from covale import make_reward_fn

reward_fn = make_reward_fn(
    method="llm",
    model="gpt-6-astra",
    registry_path="atlas_registry/registry.json",
    concurrency=4,
)

rewards = reward_fn(
    completions=generated_descriptions,
    ground_truth=reference_descriptions,
)
```

The callable accepts plain strings or conversational message lists and
returns one Dice reward per completion.
`benchmark.rewards.benchmark_reward` reports reward latency and mean reward.

```python
from benchmark.rewards import benchmark_reward

timing = benchmark_reward(
    reward_fn,
    completions=generated_descriptions,
    references=reference_descriptions,
    repeats=3,
)
print(timing)
```

## Adding a new atlas primitive

See [atlas_registry/README.md](atlas_registry/README.md) to add atlas
primitives. 