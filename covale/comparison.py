from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any, Literal

import numpy as np

Metric = Callable[[Sequence[str], Sequence[str]], object]
SignificanceTest = Literal["bootstrap", "approximate_randomization"]


def metric_value(result: object) -> float:
    if isinstance(result, Real) and not isinstance(result, bool):
        value = float(result)
    elif isinstance(result, Mapping):
        candidate = (
            result["score"]
            if "score" in result
            else next(iter(result.values()), None)
        )
        if not isinstance(candidate, Real) or isinstance(candidate, bool):
            raise ValueError("Metric mapping must contain a numeric value.")
        value = float(candidate)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if (
            not result
            or not isinstance(result[0], Real)
            or isinstance(result[0], bool)
        ):
            raise ValueError("Metric sequence must start with a numeric value.")
        value = float(result[0])
    else:
        raise ValueError("Metric must return a number, mapping, or sequence.")

    if not np.isfinite(value):
        raise ValueError("Metric score must be finite.")
    return value


def compare_systems(
    systems: Mapping[str, Sequence[str]],
    metrics: Mapping[str, Metric],
    references: Sequence[str],
    n_samples: int = 10_000,
    *,
    significance_level: float = 0.05,
    random_seed: int | None = 0,
    test: SignificanceTest = "bootstrap",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    if len(systems) < 2:
        raise ValueError("At least two systems are required.")
    if not metrics:
        raise ValueError("At least one metric is required.")
    if not references:
        raise ValueError("References must not be empty.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1.")
    if test not in {"bootstrap", "approximate_randomization"}:
        raise ValueError(f"Unknown significance test: {test}")

    for name, outputs in systems.items():
        if len(outputs) != len(references):
            raise ValueError(
                f"System '{name}' has {len(outputs)} outputs for "
                f"{len(references)} references."
            )

    system_names = list(systems)
    baseline_name = system_names[0]
    cached: dict[str, dict[str, np.ndarray]] = {
        system_name: {} for system_name in system_names
    }
    scores: dict[str, dict[str, float]] = {
        system_name: {} for system_name in system_names
    }

    for system_name, outputs in systems.items():
        for metric_name, metric in metrics.items():
            pair_scores = np.array(
                [
                    metric_value(metric([candidate], [reference]))
                    for candidate, reference in zip(
                        outputs, references, strict=True
                    )
                ],
                dtype=float,
            )
            cached[system_name][metric_name] = pair_scores
            scores[system_name][metric_name] = float(pair_scores.mean())

    rng = np.random.default_rng(random_seed)
    signatures: list[dict[str, Any]] = []
    confidence = [50 * significance_level, 100 - 50 * significance_level]

    for system_name in system_names[1:]:
        for metric_name in metrics:
            differences = (
                cached[system_name][metric_name]
                - cached[baseline_name][metric_name]
            )
            delta = float(differences.mean())
            if test == "bootstrap":
                sampled_deltas = np.empty(n_samples, dtype=float)
                for sample in range(n_samples):
                    indices = rng.integers(
                        0, len(references), len(references)
                    )
                    sampled_deltas[sample] = differences[indices].mean()

                non_positive = (
                    np.count_nonzero(sampled_deltas <= 0) + 1
                ) / (n_samples + 1)
                non_negative = (
                    np.count_nonzero(sampled_deltas >= 0) + 1
                ) / (n_samples + 1)
                p_value = min(1.0, 2 * min(non_positive, non_negative))
                lower, upper = np.percentile(sampled_deltas, confidence)
                confidence_interval: list[float] | None = [
                    float(lower),
                    float(upper),
                ]
            else:
                randomized = np.empty(n_samples, dtype=float)
                for sample in range(n_samples):
                    signs = rng.choice((-1.0, 1.0), size=len(references))
                    randomized[sample] = (differences * signs).mean()
                extreme = np.count_nonzero(
                    np.abs(randomized) >= abs(delta)
                )
                p_value = (extreme + 1) / (n_samples + 1)
                confidence_interval = None

            scores[system_name][f"{metric_name}_pvalue"] = p_value
            signatures.append(
                {
                    "baseline": baseline_name,
                    "system": system_name,
                    "metric": metric_name,
                    "delta": delta,
                    "confidence_interval": confidence_interval,
                    "p_value": p_value,
                    "significant": p_value < significance_level,
                    "test": test,
                    "n_samples": n_samples,
                    "random_seed": random_seed,
                }
            )

    return signatures, scores
