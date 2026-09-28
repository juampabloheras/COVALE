import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from covale.evaluate import covale, evaluate, evaluate_batch
from covale.localize import deep_agent
from covale.localize.openai_client import ModelResponseError
from covale.localize import LocalizationError, execute_expression


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    masks = tmp_path / "masks"
    masks.mkdir()
    affine = np.eye(4)
    nib.save(
        nib.Nifti1Image(np.array([1, 1, 0], dtype=np.uint8), affine),
        masks / "a.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.array([0, 1, 1], dtype=np.uint8), affine),
        masks / "b.nii.gz",
    )
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "atlas": "test",
                "space": "test",
                "regions": {
                    "region a": {
                        "mask": "masks/a.nii.gz",
                        "aliases": ["first region"],
                    },
                    "region b": {
                        "mask": "masks/b.nii.gz",
                        "aliases": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def region(name: str) -> dict[str, str]:
    return {"op": "region", "name": name}


def test_direct_region_lookup(registry: Path) -> None:
    result = execute_expression(region("first region"), registry)
    np.testing.assert_array_equal(result.data, [True, True, False])


def test_union_expression(registry: Path) -> None:
    expression = {"op": "union", "args": [region("region a"), region("region b")]}
    result = execute_expression(expression, registry)
    np.testing.assert_array_equal(result.data, [True, True, True])


def test_intersection_expression(registry: Path) -> None:
    expression = {
        "op": "intersection",
        "args": [region("region a"), region("region b")],
    }
    result = execute_expression(expression, registry)
    np.testing.assert_array_equal(result.data, [False, True, False])


def test_difference_expression(registry: Path) -> None:
    expression = {
        "op": "difference",
        "args": [region("region a"), region("region b")],
    }
    result = execute_expression(expression, registry)
    np.testing.assert_array_equal(result.data, [True, False, False])


def test_unknown_region(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="Unknown atlas region"):
        execute_expression(region("missing"), registry)


def test_unknown_operator(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="Unknown mask operator"):
        execute_expression({"op": "eval", "args": []}, registry)


def test_malformed_expression(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="requires an args list"):
        execute_expression({"op": "union"}, registry)


def test_unresolved_expression(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="could not be localized"):
        execute_expression({"op": "unresolved"}, registry)


def test_registry_path_cannot_escape(registry: Path) -> None:
    contents = json.loads(registry.read_text(encoding="utf-8"))
    contents["regions"]["escape"] = {"mask": "../outside.nii.gz", "aliases": []}
    registry.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(LocalizationError, match="inside atlas_registry"):
        execute_expression(region("escape"), registry)


def test_evaluate_localizes_descriptions_independently(registry: Path) -> None:
    localized: list[str] = []

    def build_expression(
        text: str, _registry: object
    ) -> dict[str, str]:
        localized.append(text)
        return region("region a" if text == "reference" else "region b")

    result = evaluate(
        "reference",
        "candidate",
        registry,
        build_expression,
    )

    assert localized == ["reference", "candidate"]
    assert result["score"] == 0.5


def test_evaluate_batch(registry: Path) -> None:
    results = evaluate_batch(
        [
            {"reference": "region a", "candidate": "first region"},
            {"reference": "region a", "candidate": "region b"},
        ],
        registry,
    )

    assert [result["score"] for result in results] == [1.0, 0.5]


def test_evaluate_batch_rejects_malformed_pairs(registry: Path) -> None:
    with pytest.raises(ValueError, match="Pair 0"):
        evaluate_batch([{"reference": "region a"}], registry)


class FakeResponses:
    def __init__(self, *outputs: str) -> None:
        self.outputs = iter(outputs)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return type("Response", (), {"output_text": next(self.outputs)})()


class FakeClient:
    def __init__(self, *outputs: str) -> None:
        self.responses = FakeResponses(*outputs)


class FakeAgent:
    def __init__(self, *outputs: str) -> None:
        self.outputs = iter(outputs)
        self.requests: list[dict[str, object]] = []

    def invoke(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return {"messages": [{"content": next(self.outputs)}]}


def test_covale_with_llm_localization(registry: Path) -> None:
    client = FakeClient(
        json.dumps(region("region a")),
        json.dumps(region("region b")),
    )

    score = covale(
        "first description",
        "second description",
        registry,
        model="test-model",
        client=client,
    )

    assert score == 0.5
    assert [request["model"] for request in client.responses.requests] == [
        "test-model",
        "test-model",
    ]


def test_covale_with_deep_agent_localization(registry: Path) -> None:
    agent = FakeAgent(
        json.dumps(region("region a")),
        json.dumps(region("region b")),
    )

    score = covale(
        "first description",
        "second description",
        registry,
        method="deep_agent",
        agent=agent,
    )

    assert score == 0.5
    assert len(agent.requests) == 2


def test_covale_with_similarity_baseline() -> None:
    client = FakeClient('{"score": 0.75}')

    score = covale(
        "left frontal lobe",
        "left frontal region",
        method="similarity",
        client=client,
    )

    assert score == 0.75


def test_covale_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown COVALE method"):
        covale("reference", "candidate", method="unknown")


def test_llm_localization_rejects_non_json(registry: Path) -> None:
    client = FakeClient("not JSON")

    with pytest.raises(ModelResponseError, match="valid JSON"):
        covale(
            "reference",
            "candidate",
            registry,
            client=client,
        )


def test_similarity_rejects_out_of_range_score() -> None:
    client = FakeClient('{"score": 1.5}')

    with pytest.raises(ValueError, match="between 0 and 1"):
        covale("reference", "candidate", method="similarity", client=client)


def test_deep_agent_model_uses_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}
    sentinel = FakeAgent()

    def fake_create_deep_agent(**kwargs: object) -> FakeAgent:
        arguments.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        deep_agent,
        "create_deep_agent",
        fake_create_deep_agent,
    )

    result = deep_agent.create_agent("test-model")

    assert result is sentinel
    assert arguments["model"] == "openai:test-model"
    assert arguments["tools"] == []


def test_evaluate_batch_with_llm_method(registry: Path) -> None:
    client = FakeClient(
        json.dumps(region("region a")),
        json.dumps(region("region a")),
        json.dumps(region("region a")),
        json.dumps(region("region b")),
    )

    results = evaluate_batch(
        [
            {"reference": "one", "candidate": "two"},
            {"reference": "three", "candidate": "four"},
        ],
        registry,
        method="llm",
        client=client,
    )

    assert [result["score"] for result in results] == [1.0, 0.5]
