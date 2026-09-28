from benchmark.rewards import benchmark_reward
from covale.rewards import completion_text, make_reward_fn


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, reference: str, candidate: str) -> float:
        self.calls.append((reference, candidate))
        return 1.0 if reference == candidate else 0.25


def test_reward_function_supports_text_and_messages() -> None:
    evaluator = FakeEvaluator()
    reward = make_reward_fn(evaluator=evaluator)

    scores = reward(
        ["same", [{"role": "assistant", "content": "candidate"}]],
        ground_truth=["same", "reference"],
    )

    assert scores == [1.0, 0.25]
    assert evaluator.calls == [
        ("same", "same"),
        ("reference", "candidate"),
    ]


def test_reward_function_supports_score_transform() -> None:
    reward = make_reward_fn(
        evaluator=FakeEvaluator(),
        score_transform=lambda score: 2 * score - 1,
    )

    assert reward(["candidate"], references=["reference"]) == [-0.5]


def test_completion_text_rejects_invalid_values() -> None:
    try:
        completion_text({"content": 42})
    except ValueError as error:
        assert "Completion" in str(error)
    else:
        raise AssertionError("Expected invalid completion content to fail.")


def test_reward_benchmark_reports_latency() -> None:
    summary = benchmark_reward(
        make_reward_fn(evaluator=FakeEvaluator()),
        completions=["a", "b"],
        references=["a", "x"],
        repeats=2,
    )

    assert summary["examples"] == 2
    assert summary["repeats"] == 2
    assert summary["mean_reward"] == 0.625
    assert summary["total_seconds"] >= 0
