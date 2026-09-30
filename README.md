# COVALE: Compositional Open-Vocabulary Anatomical Localization Evaluation

Text-based metrics can tell whether two anatomical descriptions use similar
words, but they often miss whether the descriptions point to the same place.
COVALE turns each description into a 3D region in an anatomical atlas, then
measures how much the regions overlap. This makes it easy to compare generated
descriptions, evaluate systems, and provide spatial feedback during model
training.


![Overview comparing COVALE with lexical and LLM-based evaluation](assets/overview.png)

## Quick start

```bash
pip install covale
```

```python
import os

from covale import COVALE

os.environ["OPENAI_API_KEY"] = "your-api-key"

score = COVALE().score(
    reference="left frontal lobe",
    candidate="left frontal region",
)
```

## Table of contents

- [Quick start](#quick-start)
- [Calculate COVALE](#calculate-covale)
- [Additional resources](#additional-resources)
- [License](#license)

## Calculate COVALE

Create an evaluator and pass equally sized lists of reference and candidate
descriptions:

```python
from covale import COVALE

covale_evaluator = COVALE(
    method="llm",
    model="gpt-6-astra",
    concurrency=4,
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

Change `model` to select an OpenAI model. Increase `concurrency` to evaluate
independent pairs in parallel.

### Choose an evaluation method

Set `method` based on how descriptions should be compared:

| Method | Description | Uses atlas masks |
|---|---|---|
| `llm` (default) | Uses one OpenAI request to select or compose regions from the full laterality-compatible registry catalog, then calculates Dice overlap | Yes |
| `deep_agent` | Conservatively checks for possible overlap, then iteratively searches, researches, composes, and validates atlas expressions only when grounding may be useful | Yes |
| `similarity` | Uses OpenAI to compare the descriptions directly as a language-only baseline; `save_volumes` is ignored with a warning | No |

Both atlas-backed methods can construct `union`, `intersection`, and
`difference` expressions when no single mask represents the requested
anatomy. `deep_agent` is bounded to 12 model/tool calls, three expression
validation attempts, three web searches, and 20 expression components.

To enable optional anatomical web research for `deep_agent`, add a Tavily key
to the same environment or ignored `.env` file as the OpenAI key:

```dotenv
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Without `TAVILY_API_KEY`, `deep_agent` still has full registry search,
browsing, metadata inspection, and expression validation tools.

Before atlas grounding, `deep_agent` makes a conservative language-model
precheck. It skips grounding and returns zero only when the regions are
classified as spatially disjoint with at least 0.9 confidence. Containment,
partial overlap, ambiguous terminology, and lower-confidence disjointness
continue through normal grounding. The precheck can request one targeted
Tavily anatomy search when external evidence would clarify specialized
terminology.

### Synthesize atlas subregions

`deep_agent` can research anatomy and create reproducible subregions when no
atlas label directly represents the request. In addition to set operations, it
can use:

- `directional_part`: retain an anterior, posterior, superior, inferior, left,
  or right fraction of a parent mask in physical RAS space
- `clip_plane`: retain one side of an arbitrary plane defined by a RAS normal
  and millimeter offset
- `dilate` and `erode`: apply Euclidean morphology using a millimeter distance

For example, an anterior-third approximation of a left hippocampus can be
represented as:

```json
{
  "op": "directional_part",
  "arg": {
    "op": "region",
    "id": "aparc-a2009s-aseg:17"
  },
  "direction": "anterior",
  "fraction": 0.3333333333
}
```

The agent can inspect a candidate mask's RAS bounds, centroid, voxel sizes, and
voxel count before choosing a plane. It validates and executes the complete
expression before returning it. Geometrically synthesized masks are explicit
approximations, not claims that the source atlas defines those subdivisions.
They flow through Dice scoring, caching, diagnostics, and `save_volumes` like
direct atlas regions.

### Save localized volumes

Pass `save_volumes` to `score()` or `evaluate_pair()` to save the exact
atlas masks used for Dice scoring:

```python
score = covale_evaluator.score(
    reference="bilateral pretectal regions",
    candidate="left pretectal region",
    save_volumes="localized_volumes",
)
```

COVALE writes one NIfTI file per localized region:

- `query_bilateral_pretectal_regions.nii.gz`, with foreground voxels labeled
  `100`
- `target_left_pretectal_region.nii.gz`, with foreground voxels labeled `200`

Names are lowercased and sanitized for filenames. Repeated names receive
suffixes such as `_2`. With `extract_findings=True`, each localized report
finding is saved separately rather than merged, so overlapping findings remain
independent volumes. The evaluator's in-memory localization cache is reused
across calls, avoiding repeated model requests and mask composition for anatomy
that has already been localized. Volume saving requires the atlas-backed `llm`
or `deep_agent` method. With `similarity`, the option is ignored and COVALE
emits a `UserWarning`.

### Evaluate reports

Add `extract_findings=True` when the inputs are sentences or reports rather
than isolated anatomical locations:

```python
report_evaluator = COVALE(
    extract_findings=True,
)
results = report_evaluator(
    refs=reference_reports,
    hyps=generated_reports,
)
```


### Inspect detailed results

Add `output_mode="detailed"` to include per-sample scores, localization
expressions, timings, and categorized failures:

```python
detailed_evaluator = COVALE(
    output_mode="detailed",
)
results = detailed_evaluator(refs=refs, hyps=hyps)
```

## Additional resources

| Guide | Description |
|---|---|
| [Configuration](docs/configuration.md) | Reproducible YAML settings, output modes, and error handling |
| [Benchmarking](docs/benchmarking.md) | Annotating JSONL experiments with COVALE scores and runtime |
| [Comparing systems](docs/comparing-systems.md) | Statistical comparison of multiple systems against shared references |
| [Publishing](docs/publishing.md) | Building and publishing releases to PyPI with GitHub Actions |
| [RL rewards](docs/rl-rewards.md) | Creating and timing TRL-compatible reward functions |
| [Onboarding a new atlas](docs/atlas-onboarding.md) | Adding atlas metadata, labels, licensing, and a labeled NIfTI volume |

## License

COVALE is available under the [MIT License](LICENSE).