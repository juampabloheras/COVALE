# COVALE: Compositional Open-Vocabulary Anatomical Localization Evaluation

Text-based metrics can tell whether two anatomical descriptions use similar
words, but they often miss whether the descriptions point to the same place.
COVALE turns each description into a 3D region in an anatomical atlas, then
measures how much the regions overlap. This makes it easy to compare generated
descriptions, evaluate systems, and provide spatial feedback during model
training.

![Overview comparing COVALE with lexical and LLM-based evaluation](assets/overview.png)

## Table of contents

- [Install and test](#install-and-test)
- [Calculate COVALE](#calculate-covale)
- [Evaluate reports](#evaluate-reports)
- [Run benchmarks](#run-benchmarks)
- [Compare systems](#compare-systems)
- [Config file](#config-file)
- [RL rewards](#rl-rewards)
- [Onboarding a new atlas](#onboarding-a-new-atlas)
- [License](#license)


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

After populating the atlas registry, pass equally sized lists of reference and
candidate descriptions:

```python
from covale import COVALE

covale_evaluator = COVALE(
    metrics=["dice"],
    method="llm",
    model="gpt-6-astra",
    registry_path="atlas_registry/registry.json",
    provider="openai",
    extract_findings=False,
    per_sample=False,
    output_mode=None,
    cache=True,
    concurrency=1,
    errors="raise",
    client=None,
    agent=None,
)
results = covale_evaluator(
    refs=[
        "left frontal lobe",
        "right temporal lobe",
    ],
    hyps=[
        "left frontal region",
        "right temporal region",
    ],
)
print(results["dice"])
```

Later evaluators only need to set options that differ from these defaults:

```python
agent_evaluator = COVALE(
    method="deep_agent",
)

baseline_evaluator = COVALE(
    method="similarity",
)
```

## Evaluate reports

Set `extract_findings=True` when the inputs are sentences or reports rather
than isolated anatomical locations:

```python
report_evaluator = COVALE(
    extract_findings=True,
    output_mode="detailed",
)
results = report_evaluator(
    refs=reference_reports,
    hyps=generated_reports,
)
```

COVALE extracts anatomically localized findings, aligns compatible findings
one to one, evaluates each matched location, and gives no credit for missing
or extra findings. Detailed output includes every match and its spatial score,
along with missing and extra findings. Location strings remain the default and
skip finding extraction.

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

Add `--extract-findings` when the JSONL rows contain report pairs.

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

extract_findings: false

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

## Onboarding a new atlas

Put the atlas metadata, labels, license, and one labeled NIfTI volume in a
single directory:

```text
NewAtlas/
├── atlas.json
├── labels.json
├── LICENSE.txt
└── NewAtlas.nii.gz
```

`atlas.json` contains the atlas ID, name, license identifier, citation, source
URL, version, role, and priority. `labels.json` maps integer voxel values to
labels and may provide stable region IDs. See the runnable example in
[`examples/atlas_onboarding`](examples/atlas_onboarding).

Run one command:

```bash
covale-onboard-atlas examples/atlas_onboarding
```

The command validates the files and voxel labels, canonicalizes the ontology,
copies the bundle under `atlas_registry/atlases/`, and updates
`atlas_registry/registry.json`.

| Mode | Use when | Behavior |
|---|---|---|
| `deterministic` | Labels are simple names | Normalizes names, extracts laterality, infers common structure types, and creates synonyms locally |
| `none` | Labels already contain canonical metadata | Preserves canonical names, synonyms, laterality, confidence, and rationale |
| `openai` | Labels need richer interpretation | Uses the consumer OpenAI API in validated chunks |

Select a mode with `--canonicalization MODE`. See
[atlas_registry/README.md](atlas_registry/README.md) for the complete file
formats and validation rules.

## License

COVALE is available under the [MIT License](LICENSE).