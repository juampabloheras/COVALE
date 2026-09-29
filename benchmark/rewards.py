from collections.abc import Callable, Sequence
from statistics import mean
from time import perf_counter


def benchmark_reward(
    reward_fn: Callable[..., list[float]],
    completions: Sequence[object],
    references: Sequence[str],
    *,
    repeats: int = 1,
) -> dict[str, float | int]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if len(completions) != len(references):
        raise ValueError("References and completions must have the same length.")

    durations = []
    scores: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        scores = reward_fn(completions, ground_truth=references)
        durations.append(perf_counter() - started)

    return {
        "examples": len(completions),
        "repeats": repeats,
        "mean_reward": mean(scores) if scores else 0.0,
        "total_seconds": sum(durations),
        "mean_repeat_seconds": mean(durations),
        "mean_example_seconds": (sum(durations) / (len(completions) * repeats) if completions else 0.0),
    }
