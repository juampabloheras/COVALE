# RL rewards

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
    registry_path="covale/atlas_registry/registry.json",
    concurrency=4,
)

rewards = reward_fn(
    completions=generated_descriptions,
    ground_truth=reference_descriptions,
)
```

The callable accepts plain strings or conversational message lists and returns
one Dice reward per completion.

`benchmark.rewards.benchmark_reward` reports reward latency and mean reward:

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

[Back to the main README](../README.md#additional-resources)
