import json
from pathlib import Path

import pytest

from covale.registry import Registry, RegistryError


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
    registry = Registry.load(
        write_registry(tmp_path / "registry.json", registry_v2())
    )

    matches = registry.search("left ACA")

    assert matches[0].region_id == "arterial:1"
    assert matches[0].match_type == "synonym"
    assert matches[0].score == 1.0


def test_search_preserves_laterality(tmp_path: Path) -> None:
    registry = Registry.load(
        write_registry(tmp_path / "registry.json", registry_v2())
    )

    matches = registry.search("left hippocampus")

    assert matches[0].region_id == "anatomical:17"
    assert all(match.laterality != "right" for match in matches)


def test_bare_name_can_return_both_lateralities(tmp_path: Path) -> None:
    registry = Registry.load(
        write_registry(tmp_path / "registry.json", registry_v2())
    )

    matches = registry.search("hippocampus")

    assert {match.region_id for match in matches[:2]} == {
        "anatomical:17",
        "anatomical:53",
    }


def test_search_supports_structured_filters(tmp_path: Path) -> None:
    registry = Registry.load(
        write_registry(tmp_path / "registry.json", registry_v2())
    )

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
    registry = Registry.load(
        write_registry(tmp_path / "registry.json", registry_v2())
    )

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
