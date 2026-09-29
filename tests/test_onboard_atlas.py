import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from covale.registry import Registry
from covale.utils.onboard_atlas import (
    AtlasOnboardingError,
    onboard_atlas,
)


def write_registry(tmp_path: Path) -> Path:
    root = tmp_path / "atlas_registry"
    root.mkdir()
    path = root / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "space": {"name": "MNI152", "resolution": "1mm"},
                "atlases": {},
                "regions": {
                    "existing:1": {
                        "canonical_name": "existing region",
                        "display_name": "existing region",
                        "source": {
                            "type": "mask",
                            "path": "masks/existing.nii.gz",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def write_bundle(
    tmp_path: Path,
    *,
    values: list[int],
    labels: object,
    atlas_id: str = "test-atlas",
) -> Path:
    bundle = tmp_path / "MNI_TestAtlas"
    bundle.mkdir()
    nib.save(
        nib.Nifti1Image(
            np.array(values, dtype=np.int16).reshape(len(values), 1, 1),
            np.eye(4),
        ),
        bundle / "TestAtlas.nii.gz",
    )
    (bundle / "atlas.json").write_text(
        json.dumps(
            {
                "id": atlas_id,
                "name": "Test Atlas",
                "role": "anatomical",
                "priority": 10,
                "source_url": "https://example.org/test-atlas",
                "version": "1.0",
                "citation": "Example et al.",
                "license": "CC-BY-4.0",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "labels.json").write_text(
        json.dumps(labels),
        encoding="utf-8",
    )
    (bundle / "LICENSE.txt").write_text(
        "Synthetic test atlas license.",
        encoding="utf-8",
    )
    return bundle


def test_onboards_directory_and_updates_registry(tmp_path: Path) -> None:
    registry_path = write_registry(tmp_path)
    bundle = write_bundle(
        tmp_path,
        values=[0, 7],
        labels={
            "7": {
                "id": "test-atlas:left-white-matter",
                "region_id": 7,
                "source_label": "left white_matter_of_forebrain",
                "canonical_name": "white matter of forebrain",
                "parent_anatomy": "forebrain",
                "laterality": "left",
                "structure_type": "white_matter",
                "is_abnormality": False,
                "synonyms": ["left white matter of forebrain"],
                "confidence": 1.0,
                "rationale": "Curated source metadata.",
            }
        },
    )

    result = onboard_atlas(
        bundle,
        registry_path=registry_path,
        canonicalization="none",
    )

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    registry = Registry.load(registry_path)
    atlas = registry.model.atlases["test-atlas"]
    region = registry.model.regions["test-atlas:left-white-matter"]
    assert result.region_count == 1
    assert result.atlas_id == "test-atlas"
    assert canonical[0]["id"] == "test-atlas:left-white-matter"
    assert canonical[0]["parent_anatomy"] == "forebrain"
    assert region.synonyms == ["left white matter of forebrain"]
    assert atlas.license == "CC-BY-4.0"
    assert atlas.citation == "Example et al."
    assert atlas.source_url == "https://example.org/test-atlas"
    assert (Path(result.atlas_root) / "LICENSE.txt").is_file()
    assert (Path(result.atlas_root) / "atlas.json").is_file()


def test_deterministic_mode_generates_default_stable_ids(
    tmp_path: Path,
) -> None:
    registry_path = write_registry(tmp_path)
    bundle = write_bundle(
        tmp_path,
        values=[0, 1],
        labels={"1": "left_hippocampus"},
    )

    result = onboard_atlas(bundle, registry_path=registry_path)

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    registry = Registry.load(registry_path)
    assert canonical[0]["id"] == "test-atlas:1"
    assert canonical[0]["canonical_name"] == "hippocampus"
    assert canonical[0]["laterality"] == "left"
    assert canonical[0]["structure_type"] == "subcortical_nucleus"
    assert "test-atlas:1" in registry.model.regions


def test_rejects_missing_bundle_files(tmp_path: Path) -> None:
    registry_path = write_registry(tmp_path)
    bundle = tmp_path / "IncompleteAtlas"
    bundle.mkdir()

    with pytest.raises(AtlasOnboardingError, match="missing required files"):
        onboard_atlas(bundle, registry_path=registry_path)


def test_rejects_multiple_volumes(tmp_path: Path) -> None:
    registry_path = write_registry(tmp_path)
    bundle = write_bundle(
        tmp_path,
        values=[0, 1],
        labels={"1": "hippocampus"},
    )
    nib.save(
        nib.Nifti1Image(
            np.array([0, 1], dtype=np.int16).reshape(2, 1, 1),
            np.eye(4),
        ),
        bundle / "Second.nii.gz",
    )

    with pytest.raises(AtlasOnboardingError, match="exactly one"):
        onboard_atlas(bundle, registry_path=registry_path)


def test_rejects_atlas_label_mismatch(tmp_path: Path) -> None:
    registry_path = write_registry(tmp_path)
    bundle = write_bundle(
        tmp_path,
        values=[0, 1, 2],
        labels={"1": "left hippocampus"},
    )

    with pytest.raises(
        AtlasOnboardingError,
        match="volume labels absent from JSON=\\[2\\]",
    ):
        onboard_atlas(bundle, registry_path=registry_path)


class FakeResponses:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return type(
            "Response",
            (),
            {"output_text": json.dumps(self.output)},
        )()


class FakeClient:
    def __init__(self, output: dict[str, object]) -> None:
        self.responses = FakeResponses(output)


def test_openai_canonicalization_preserves_ids_and_source_label(
    tmp_path: Path,
) -> None:
    registry_path = write_registry(tmp_path)
    bundle = write_bundle(
        tmp_path,
        values=[0, 1],
        labels={
            "1": {
                "id": "test-atlas:left-cortex",
                "label": "ctx-lh-test",
            }
        },
    )
    client = FakeClient(
        {
            "regions": [
                {
                    "region_id": 1,
                    "source_label": "changed by model",
                    "canonical_name": "test cortex",
                    "laterality": "left",
                    "structure_type": "cortical_region",
                    "is_abnormality": False,
                    "synonyms": ["left test cortex"],
                    "confidence": 0.9,
                    "rationale": "Expanded compact label.",
                }
            ]
        }
    )

    result = onboard_atlas(
        bundle,
        registry_path=registry_path,
        canonicalization="openai",
        model="test-model",
        client=client,
    )

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    assert canonical[0]["id"] == "test-atlas:left-cortex"
    assert canonical[0]["source_label"] == "ctx-lh-test"
    assert client.responses.requests[0]["model"] == "test-model"
