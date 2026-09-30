import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from covale import COVALE, covale, evaluate, evaluate_batch
from covale.localize import deep_agent, overlap
from covale.localize.openai_client import ModelResponseError
from covale.localize import LocalizationError, execute_expression
from covale.localize.registry_tools import RegistryLocalizationTools
from covale.registry import Registry


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


def test_directional_part_uses_ras_extent(registry: Path) -> None:
    result = execute_expression(
        {
            "op": "directional_part",
            "arg": region("region a"),
            "direction": "right",
            "fraction": 0.5,
        },
        registry,
    )

    np.testing.assert_array_equal(result.data, [False, True, False])


def test_clip_plane_uses_ras_millimeters(registry: Path) -> None:
    result = execute_expression(
        {
            "op": "clip_plane",
            "arg": {
                "op": "union",
                "args": [region("region a"), region("region b")],
            },
            "normal": [1, 0, 0],
            "offset_mm": 1.5,
            "side": "positive",
        },
        registry,
    )

    np.testing.assert_array_equal(result.data, [False, False, True])


def test_morphology_uses_millimeter_distance(registry: Path) -> None:
    dilated = execute_expression(
        {
            "op": "dilate",
            "arg": region("region a"),
            "distance_mm": 1,
        },
        registry,
    )
    eroded = execute_expression(
        {
            "op": "erode",
            "arg": region("region a"),
            "distance_mm": 1,
        },
        registry,
    )

    np.testing.assert_array_equal(dilated.data, [True, True, True])
    np.testing.assert_array_equal(eroded.data, [True, False, False])


def test_registry_tools_search_and_validate_composition(registry: Path) -> None:
    tools = RegistryLocalizationTools(Registry.load(registry))

    matches = tools.search_regions("first region")
    validation = tools.validate_expression(
        {
            "op": "union",
            "args": [
                {"op": "region", "id": "legacy:region-a"},
                {"op": "region", "id": "legacy:region-b"},
            ],
        }
    )

    assert matches[0]["id"] == "legacy:region-a"
    assert validation == {
        "valid": True,
        "expression": {
            "op": "union",
            "args": [
                {"op": "region", "id": "legacy:region-a"},
                {"op": "region", "id": "legacy:region-b"},
            ],
        },
        "component_ids": ["legacy:region-a", "legacy:region-b"],
        "voxel_count": 3,
        "shape": [3],
    }


def test_registry_tools_inspect_primitive_geometry(registry: Path) -> None:
    tools = RegistryLocalizationTools(Registry.load(registry))
    expression = {
        "op": "directional_part",
        "arg": {"op": "region", "id": "legacy:region-a"},
        "direction": "right",
        "fraction": 0.5,
    }

    validation = tools.validate_expression(expression)
    inspected = tools.inspect_expression_geometry(expression)

    assert validation["valid"] is True
    assert validation["component_ids"] == ["legacy:region-a"]
    assert inspected["valid"] is True
    assert inspected["centroid_ras_mm"] == [1.0, 0.0, 0.0]
    assert inspected["bounds_ras_mm"] == {
        "minimum": [1.0, 0.0, 0.0],
        "maximum": [1.0, 0.0, 0.0],
    }


def test_registry_tools_report_invalid_expression(registry: Path) -> None:
    tools = RegistryLocalizationTools(Registry.load(registry))

    result = tools.validate_expression(
        {"op": "region", "id": "legacy:missing"}
    )

    assert result["valid"] is False
    assert "Unknown registry region ID" in result["error"]


def test_registry_tools_limit_expression_components(registry: Path) -> None:
    tools = RegistryLocalizationTools(Registry.load(registry))

    result = tools.validate_expression(
        {
            "op": "union",
            "args": [
                {"op": "region", "id": "legacy:region-a"}
                for _ in range(21)
            ],
        }
    )

    assert result["valid"] is False
    assert "at most 20" in result["error"]


def test_unknown_region(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="Unknown atlas region"):
        execute_expression(region("missing"), registry)


def test_unknown_operator(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="Invalid mask expression"):
        execute_expression({"op": "eval", "args": []}, registry)


def test_malformed_expression(registry: Path) -> None:
    with pytest.raises(LocalizationError, match="Invalid mask expression"):
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

    def build_expression(text: str, _registry: object) -> dict[str, str]:
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


def test_evaluate_saves_labeled_query_and_target_volumes(
    registry: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "volumes"

    result = evaluate(
        "region a",
        "region b",
        registry,
        save_volumes=output_dir,
    )

    query_path = output_dir / "query_region_a.nii.gz"
    target_path = output_dir / "target_region_b.nii.gz"
    np.testing.assert_array_equal(
        np.asarray(nib.load(query_path).dataobj),
        [100, 100, 0],
    )
    np.testing.assert_array_equal(
        np.asarray(nib.load(target_path).dataobj),
        [0, 200, 200],
    )
    assert result["diagnostics"]["saved_volumes"] == {
        "query": str(query_path.resolve()),
        "target": str(target_path.resolve()),
    }


def test_covale_reuses_localization_cache_when_saving(
    registry: Path,
    tmp_path: Path,
) -> None:
    client = FakeClient(
        json.dumps(region("region a")),
        json.dumps(region("region b")),
        json.dumps(region("region b")),
    )
    evaluator = COVALE(
        registry_path=registry,
        client=client,
    )

    evaluator.score(
        "shared query description",
        "first target description",
        save_volumes=tmp_path / "first",
    )
    evaluator.score(
        "shared query description",
        "second target description",
        save_volumes=tmp_path / "second",
    )

    assert len(client.responses.requests) == 3
    assert (tmp_path / "second/query_shared_query_description.nii.gz").is_file()


def test_similarity_warns_and_ignores_volume_saving(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="does not produce atlas masks"):
        result = evaluate(
            "query",
            "target",
            method="similarity",
            client=FakeClient('{"score": 1.0}'),
            save_volumes=tmp_path,
        )

    assert result["score"] == 1.0
    assert list(tmp_path.iterdir()) == []


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
    client = FakeClient(
        '{"confidently_disjoint": false, "confidence": 0.95}'
    )

    score = covale(
        "first description",
        "second description",
        registry,
        method="deep_agent",
        agent=agent,
        client=client,
    )

    assert score == 0.5
    assert len(agent.requests) == 2
    assert len(client.responses.requests) == 1


def test_deep_agent_skips_grounding_for_confidently_disjoint_regions(
    registry: Path,
) -> None:
    agent = FakeAgent()
    client = FakeClient(
        '{"confidently_disjoint": true, "confidence": 0.99}'
    )

    result = evaluate(
        "left hippocampus",
        "right frontal pole",
        registry,
        method="deep_agent",
        agent=agent,
        client=client,
    )

    assert result["score"] == 0.0
    assert result["diagnostics"]["grounding_skipped"] is True
    assert result["diagnostics"]["overlap_precheck"] == {
        "confidently_disjoint": True,
        "confidence": 0.99,
    }
    assert agent.requests == []


def test_deep_agent_grounds_when_disjointness_is_uncertain(
    registry: Path,
) -> None:
    agent = FakeAgent(
        json.dumps(region("region a")),
        json.dumps(region("region b")),
    )
    client = FakeClient(
        '{"confidently_disjoint": true, "confidence": 0.5}'
    )

    result = evaluate(
        "first description",
        "second description",
        registry,
        method="deep_agent",
        agent=agent,
        client=client,
    )

    assert result["score"] == 0.5
    assert result["diagnostics"]["grounding_skipped"] is False
    assert len(agent.requests) == 2


def test_overlap_precheck_can_research_ambiguous_anatomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(
        json.dumps(
            {
                "confidently_disjoint": False,
                "confidence": 0.4,
                "research_query": "specialized anatomy relationship",
            }
        ),
        json.dumps(
            {
                "confidently_disjoint": True,
                "confidence": 0.97,
                "research_query": None,
            }
        ),
    )
    searches = []

    def fake_search(
        query: str,
        *,
        api_key: str,
        max_results: int = 5,
    ) -> dict[str, object]:
        searches.append((query, api_key, max_results))
        return {"results": [{"title": "Anatomy source"}]}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(overlap, "tavily_search", fake_search)

    result = overlap.precheck_overlap(
        "first anatomy",
        "second anatomy",
        client=client,
        model="test-model",
    )

    assert result.confidently_disjoint is True
    assert searches == [
        ("specialized anatomy relationship", "test-key", 5)
    ]
    second_payload = json.loads(client.responses.requests[1]["input"])
    assert second_payload["web_research"]["results"] == [
        {"title": "Anatomy source"}
    ]


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

    with pytest.raises(ValueError, match="less than or equal to 1"):
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
        "deepagents.create_deep_agent",
        fake_create_deep_agent,
    )

    result = deep_agent.create_agent("test-model")

    assert result is sentinel
    assert arguments["model"] == "openai:test-model"
    assert arguments["tools"] == []
    assert len(arguments["middleware"]) == 3


def test_deep_agent_receives_registry_tools(
    registry: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments: dict[str, object] = {}
    sentinel = FakeAgent()

    def fake_create_deep_agent(**kwargs: object) -> FakeAgent:
        arguments.update(kwargs)
        return sentinel

    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setattr(
        "deepagents.create_deep_agent",
        fake_create_deep_agent,
    )

    result = deep_agent.create_agent(
        "test-model",
        Registry.load(registry),
    )

    assert result is sentinel
    assert [tool.__name__ for tool in arguments["tools"]] == [
        "search_regions",
        "list_regions",
        "get_region",
        "inspect_expression_geometry",
        "validate_atlas_expression",
    ]


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
