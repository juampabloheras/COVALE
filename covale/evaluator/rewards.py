from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import Any

from covale.evaluator import COVALE

ScoreTransform = Callable[[float], float]


def completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        if not completion:
            raise ValueError("Conversational completion must not be empty.")
        return completion_text(completion[-1])
    raise ValueError("Completion must be text or a message with string content.")


def make_reward_fn(
    evaluator: COVALE | None = None,
    *,
    score_transform: ScoreTransform | None = None,
    **evaluator_options: Any,
) -> Callable[..., list[float]]:
    if evaluator is not None and evaluator_options:
        raise ValueError("Pass an evaluator or evaluator options, not both.")
    evaluator = evaluator or COVALE(
        metrics=["dice"],
        **evaluator_options,
    )

    def reward(
        completions: Sequence[object],
        ground_truth: Sequence[str] | str | None = None,
        *,
        references: Sequence[str] | str | None = None,
        **_: object,
    ) -> list[float]:
        selected_references = ground_truth if ground_truth is not None else references
        if selected_references is None:
            raise ValueError("Reward calls require ground_truth or references.")
        if isinstance(selected_references, str):
            selected_references = [selected_references] * len(completions)
        if len(selected_references) != len(completions):
            raise ValueError("References and completions must have the same length.")

        scores = [
            evaluator.score(reference, completion_text(completion))
            for reference, completion in zip(
                selected_references,
                completions,
                strict=True,
            )
        ]
        if score_transform is not None:
            scores = [float(score_transform(score)) for score in scores]
        return scores

    return reward
