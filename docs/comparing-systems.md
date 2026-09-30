# Comparing systems

Compare multiple systems against shared reference descriptions:

```python
from covale import COVALE, compare_systems

covale_evaluator = COVALE()
signatures, scores = compare_systems(
    systems={
        "baseline": baseline_descriptions,
        "comparison_1": improved_descriptions,
        "comparison_2": improved_descriptions,
        "proposed": improved_descriptions,
    },
    metrics={
        "dice": lambda hyps, refs: covale_evaluator(refs, hyps)["dice"],
    },
    references=reference_descriptions,
    n_samples=10_000,
    test="bootstrap",
)
```

`scores` contains system means and p-values. `signatures` contains deltas,
confidence intervals, significance flags, and reproducibility settings for
each comparison.

Set `test="approximate_randomization"` for a paired randomization test instead
of paired bootstrap.

[Back to the main README](../README.md#additional-resources)
