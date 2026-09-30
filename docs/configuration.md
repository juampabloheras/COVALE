# Configuration

Use YAML to keep evaluator settings reproducible:

```yaml
metrics:
  - dice:
      method: llm
      provider: openai
      model_name: gpt-6-astra
      registry_path: ../covale/atlas_registry/registry.json
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
pairs in the result. See [`examples/config.yaml`](../examples/config.yaml) for
a complete example.

[Back to the main README](../README.md#additional-resources)
