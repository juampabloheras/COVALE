import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from covale.localize import LocalizationError, execute_expression
from covale.localize.openai_client import ModelResponseError
from covale.registry import Registry, RegistryError, RegistryResolver
from covale.registry.migrate import MigrationError, migrate_registry


def write_registry(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def registry_v2() -> dict[str, object]:
    return {
        "schema_version": 2,
        "space": {"name": "MNI152", "resolution": "1mm"},
        "atlases": {
            "arterial": {
                "name": "Arterial Atlas",
                "volume": "atlases/arterial.nii.gz",
                "role": "vascular",
                "priority": 20,
            },
            "anatomical": {
                "name": "Anatomical Atlas",
                "volume": "atlases/anatomical.nii.gz",
                "priority": 10,
            },
        },
        "regions": {
            "arterial:1": {
                "canonical_name": "anterior cerebral artery",
                "display_name": "left anterior cerebral artery",
                "source_label": "anterior cerebral artery left",
                "laterality": "left",
                "structure_type": "vascular",
                "synonyms": ["left ACA", "ACA left"],
                "source": {
                    "type": "labels",
                    "atlas": "arterial",
                    "values": [1],
                },
            },
            "anatomical:17": {
                "canonical_name": "hippocampus",
                "display_name": "left hippocampus",
                "laterality": "left",
                "structure_type": "subcortical_nucleus",
                "synonyms": ["left hippocampal formation"],
                "source": {
                    "type": "labels",
                    "atlas": "anatomical",
                    "values": [17],
                },
            },
            "anatomical:53": {
                "canonical_name": "hippocampus",
                "display_name": "right hippocampus",
                "laterality": "right",
                "structure_type": "subcortical_nucleus",
                "synonyms": ["right hippocampal formation"],
                "source": {
                    "type": "labels",
                    "atlas": "anatomical",
                    "values": [53],
                },
            },
        },
    }


def test_loads_registry_v2_and_searches_synonyms(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    matches = registry.search("left ACA")

    assert matches[0].region_id == "arterial:1"
    assert matches[0].match_type == "synonym"
    assert matches[0].score == 1.0


def test_search_preserves_laterality(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    matches = registry.search("left hippocampus")

    assert matches[0].region_id == "anatomical:17"
    assert all(match.laterality != "right" for match in matches)


def test_bare_name_can_return_both_lateralities(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    matches = registry.search("hippocampus")

    assert {match.region_id for match in matches[:2]} == {
        "anatomical:17",
        "anatomical:53",
    }


def test_bilateral_query_includes_unilateral_components(
    tmp_path: Path,
) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    matches = registry.search("bilateral hippocampi")

    assert {match.region_id for match in matches} == {
        "anatomical:17",
        "anatomical:53",
        "arterial:1",
    }


def test_search_supports_structured_filters(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    matches = registry.search(
        "hippocampus",
        laterality="right",
        structure_type="subcortical_nucleus",
        atlas="anatomical",
    )

    assert [match.region_id for match in matches] == ["anatomical:53"]


def test_loads_v1_registry_as_v2(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path / "registry.json",
        {
            "atlas": "example",
            "space": "MNI152",
            "regions": {
                "left frontal lobe": {
                    "mask": "masks/left-frontal.nii.gz",
                    "aliases": ["left frontal region"],
                }
            },
        },
    )

    registry = Registry.load(path)

    assert registry.model.schema_version == 2
    assert registry.search("left frontal region")[0].region_id == (
        "legacy:left-frontal-lobe"
    )


def test_rejects_unknown_atlas_reference(tmp_path: Path) -> None:
    payload = registry_v2()
    payload["regions"]["arterial:1"]["source"]["atlas"] = "missing"

    with pytest.raises(RegistryError, match="unknown atlas"):
        Registry.load(write_registry(tmp_path / "registry.json", payload))


def test_compact_catalog_contains_stable_ids(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))

    catalog = registry.compact_catalog("left ACA", limit=1)

    assert catalog == [
        {
            "id": "arterial:1",
            "name": "left anterior cerebral artery",
            "laterality": "left",
            "structure_type": "vascular",
            "synonyms": ["left ACA", "ACA left"],
        }
    ]


class FakeResponses:
    def __init__(self, *outputs: dict[str, object]) -> None:
        self.outputs = iter(outputs)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return type(
            "Response",
            (),
            {"output_text": json.dumps(next(self.outputs))},
        )()


class FakeClient:
    def __init__(self, *outputs: dict[str, object]) -> None:
        self.responses = FakeResponses(*outputs)


def test_resolver_exact_synonym_bypasses_openai(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))
    client = FakeClient()

    expression = RegistryResolver(
        registry,
        client=client,
        model="test-model",
    ).resolve("left ACA")

    assert expression == {"op": "region", "id": "arterial:1"}
    assert client.responses.requests == []


def test_resolver_uses_compact_candidates_for_composition(
    tmp_path: Path,
) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))
    client = FakeClient(
        {
            "op": "union",
            "args": [
                {"op": "region", "id": "anatomical:17"},
                {"op": "region", "id": "anatomical:53"},
            ],
        }
    )

    expression = RegistryResolver(
        registry,
        client=client,
        model="test-model",
    ).resolve("bilateral hippocampi")

    assert expression["op"] == "union"
    request = client.responses.requests[0]
    payload = json.loads(request["input"])
    assert "atlas_registry" not in payload
    assert {candidate["id"] for candidate in payload["candidate_regions"]} == {
        "anatomical:17",
        "anatomical:53",
        "arterial:1",
    }


def test_resolver_rejects_id_outside_candidates(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))
    client = FakeClient({"op": "region", "id": "unknown:1"})

    with pytest.raises(ModelResponseError, match="outside candidate catalog"):
        RegistryResolver(
            registry,
            client=client,
            model="test-model",
        ).resolve("unlisted anatomy")


def test_resolver_rejects_conflicting_laterality(tmp_path: Path) -> None:
    registry = Registry.load(write_registry(tmp_path / "registry.json", registry_v2()))
    client = FakeClient(
        {"op": "region", "id": "anatomical:53"},
    )

    with pytest.raises(ModelResponseError, match="outside candidate catalog"):
        RegistryResolver(
            registry,
            client=client,
            model="test-model",
        ).resolve("left hippocampal area")


def write_labeled_registry(
    tmp_path: Path,
    values: list[int],
) -> Path:
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    nib.save(
        nib.Nifti1Image(
            np.array(values, dtype=np.int16).reshape(len(values), 1, 1),
            np.eye(4),
        ),
        atlas_dir / "arterial.nii.gz",
    )
    payload = registry_v2()
    payload["regions"] = {
        "arterial:1": payload["regions"]["arterial:1"],
    }
    return write_registry(tmp_path / "registry.json", payload)


def test_executes_stable_id_from_labeled_atlas(tmp_path: Path) -> None:
    path = write_labeled_registry(tmp_path, [0, 1, 2])

    result = execute_expression(
        {"op": "region", "id": "arterial:1"},
        path,
    )

    np.testing.assert_array_equal(result.data.ravel(), [False, True, False])


def test_labeled_region_can_combine_multiple_values(tmp_path: Path) -> None:
    path = write_labeled_registry(tmp_path, [0, 1, 2])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["regions"]["arterial:1"]["source"]["values"] = [1, 2]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = execute_expression(
        {"op": "region", "id": "arterial:1"},
        path,
    )

    np.testing.assert_array_equal(result.data.ravel(), [False, True, True])


def test_rejects_missing_atlas_label(tmp_path: Path) -> None:
    path = write_labeled_registry(tmp_path, [0, 1, 2])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["regions"]["arterial:1"]["source"]["values"] = [99]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LocalizationError, match="missing atlas labels"):
        execute_expression(
            {"op": "region", "id": "arterial:1"},
            path,
        )


def test_rejects_atlas_path_outside_registry(tmp_path: Path) -> None:
    path = write_labeled_registry(tmp_path, [0, 1, 2])
    outside = tmp_path.parent / "outside.nii.gz"
    nib.save(
        nib.Nifti1Image(np.array([0, 1], dtype=np.int16), np.eye(4)),
        outside,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["atlases"]["arterial"]["volume"] = "../outside.nii.gz"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LocalizationError, match="inside atlas_registry"):
        execute_expression(
            {"op": "region", "id": "arterial:1"},
            path,
        )


def write_migration_source(
    tmp_path: Path,
    *,
    region_id: int = 1,
) -> Path:
    source = tmp_path / "source"
    atlas = source / "MNI_TestAtlas"
    volumes = atlas / "volumes"
    volumes.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(
            np.array([0, 1, 2], dtype=np.int16).reshape(3, 1, 1),
            np.eye(4),
        ),
        volumes / "TestAtlas.nii.gz",
    )
    (atlas / "TestAtlas_canonical_names.json").write_text(
        json.dumps(
            [
                {
                    "region_id": region_id,
                    "source_label": "hippocampus left",
                    "canonical_name": "hippocampus",
                    "parent_anatomy": "temporal lobe",
                    "laterality": "left",
                    "structure_type": "subcortical_nucleus",
                    "is_abnormality": False,
                    "synonyms": ["left hippocampal formation"],
                    "confidence": 0.99,
                    "rationale": "Canonical test record.",
                    "atlas_sources": ["TestAtlas"],
                    "preferred_atlas": "TestAtlas",
                }
            ]
        ),
        encoding="utf-8",
    )
    return source


def test_migrates_canonical_atlas_bundle(tmp_path: Path) -> None:
    source = write_migration_source(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "TestAtlas": {
                    "name": "Test Atlas",
                    "license": "CC-BY-4.0",
                    "citation": "Example et al.",
                    "role": "anatomical",
                    "priority": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    report = migrate_registry(
        source,
        output,
        space="MNI152",
        resolution="1mm",
        metadata_path=metadata,
    )

    assert report.atlas_count == 1
    assert report.region_count == 1
    assert report.warnings == []
    registry = Registry.load(output / "registry.json")
    region = registry.model.regions["testatlas:1"]
    assert region.display_name == "left hippocampus"
    assert region.parent_anatomy == "temporal lobe"
    assert region.synonyms == ["left hippocampal formation"]
    assert registry.model.atlases["testatlas"].license == "CC-BY-4.0"
    assert (output / "migration_report.json").is_file()
    result = execute_expression(
        {"op": "region", "id": "testatlas:1"},
        output / "registry.json",
    )
    np.testing.assert_array_equal(result.data.ravel(), [False, True, False])


def test_migration_reports_missing_provenance(tmp_path: Path) -> None:
    source = write_migration_source(tmp_path)

    report = migrate_registry(source, tmp_path / "output", space="MNI152")

    assert report.warnings == [
        "Atlas 'testatlas' has no license metadata.",
        "Atlas 'testatlas' has no citation metadata.",
    ]


def test_migration_rejects_missing_volume_label(tmp_path: Path) -> None:
    source = write_migration_source(tmp_path, region_id=99)

    with pytest.raises(MigrationError, match="missing from atlas"):
        migrate_registry(source, tmp_path / "output", space="MNI152")
