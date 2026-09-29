import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from covale.utils.onboard_atlas import (
    AtlasOnboardingError,
    onboard_atlas,
)


def write_volume(path: Path, values: list[int]) -> Path:
    nib.save(
        nib.Nifti1Image(
            np.array(values, dtype=np.int16).reshape(len(values), 1, 1),
            np.eye(4),
        ),
        path,
    )
    return path


def test_onboards_tagged_dictionary_labels_without_recanonicalizing(
    tmp_path: Path,
) -> None:
    volume = write_volume(tmp_path / "NextBrain.nii.gz", [0, 7])
    labels = tmp_path / "NextBrain_labels.json"
    labels.write_text(
        json.dumps(
            {
                "7": {
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
            }
        ),
        encoding="utf-8",
    )

    result = onboard_atlas(
        "MNI_NextBrain",
        volume,
        tmp_path / "atlases",
        labels_json=labels,
        canonicalization="none",
    )

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    assert result.region_count == 1
    assert canonical[0]["canonical_name"] == "white matter of forebrain"
    assert canonical[0]["parent_anatomy"] == "forebrain"
    assert canonical[0]["synonyms"] == ["left white matter of forebrain"]
    assert canonical[0]["atlas_sources"] == ["NextBrain"]
    assert canonical[0]["preferred_atlas"] == "NextBrain"
    assert Path(result.labels).read_text(encoding="utf-8") == labels.read_text(
        encoding="utf-8"
    )


def test_onboards_with_deterministic_canonicalization(tmp_path: Path) -> None:
    volume = write_volume(tmp_path / "anatomical.nii.gz", [0, 1])
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps({"1": "left_hippocampus"}),
        encoding="utf-8",
    )

    result = onboard_atlas(
        "MNI_Anatomical",
        volume,
        tmp_path / "atlases",
        labels_json=labels,
    )

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    assert canonical[0]["canonical_name"] == "hippocampus"
    assert canonical[0]["laterality"] == "left"
    assert canonical[0]["structure_type"] == "subcortical_nucleus"
    assert "left hippocampus" in canonical[0]["synonyms"]


def test_generates_labels_for_binary_mask(tmp_path: Path) -> None:
    volume = write_volume(tmp_path / "brainstem.nii.gz", [0, 1])

    result = onboard_atlas(
        "MNI_Brainstem",
        volume,
        tmp_path / "atlases",
    )

    assert json.loads(Path(result.labels).read_text(encoding="utf-8")) == {
        "1": "brainstem"
    }
    assert result.region_count == 1


def test_requires_labels_for_multilabel_volume(tmp_path: Path) -> None:
    volume = write_volume(tmp_path / "anatomical.nii.gz", [0, 1, 2])

    with pytest.raises(AtlasOnboardingError, match="labels_json is required"):
        onboard_atlas(
            "MNI_Anatomical",
            volume,
            tmp_path / "atlases",
        )


def test_rejects_atlas_label_mismatch(tmp_path: Path) -> None:
    volume = write_volume(tmp_path / "anatomical.nii.gz", [0, 1, 2])
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"1": "left hippocampus"}), encoding="utf-8")

    with pytest.raises(
        AtlasOnboardingError,
        match="volume labels absent from JSON=\\[2\\]",
    ):
        onboard_atlas(
            "MNI_Anatomical",
            volume,
            tmp_path / "atlases",
            labels_json=labels,
        )


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


def test_openai_canonicalization_preserves_source_label(tmp_path: Path) -> None:
    volume = write_volume(tmp_path / "anatomical.nii.gz", [0, 1])
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"1": "ctx-lh-test"}), encoding="utf-8")
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
        "MNI_Anatomical",
        volume,
        tmp_path / "atlases",
        labels_json=labels,
        canonicalization="openai",
        model="test-model",
        client=client,
    )

    canonical = json.loads(Path(result.canonical_json).read_text(encoding="utf-8"))
    assert canonical[0]["source_label"] == "ctx-lh-test"
    assert client.responses.requests[0]["model"] == "test-model"
