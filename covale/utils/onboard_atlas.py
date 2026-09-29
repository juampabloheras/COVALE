import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from pydantic import Field, ValidationError, model_validator

from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)
from covale.models import StrictModel
from covale.prompts import ATLAS_ONBOARDING_PROMPT
from covale.registry.masks import load_label_volume
from covale.registry.models import Laterality

Canonicalization = Literal["none", "deterministic", "openai"]


class AtlasOnboardingError(ValueError):
    pass


@dataclass(frozen=True)
class LabelEntry:
    region_id: int
    source_label: str
    original: Mapping[str, Any] | None = None


class CanonicalLabel(StrictModel):
    region_id: int = Field(gt=0)
    source_label: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    parent_anatomy: str | None = None
    laterality: Laterality = "unknown"
    structure_type: str = Field(default="unknown", min_length=1)
    is_abnormality: bool = False
    synonyms: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    rationale: str = ""
    atlas_sources: list[str] = Field(default_factory=list)
    preferred_atlas: str | None = None

    @model_validator(mode="after")
    def normalize_synonyms(self) -> "CanonicalLabel":
        synonyms = []
        seen = set()
        for value in self.synonyms:
            value = value.strip()
            folded = value.casefold()
            if value and folded not in seen:
                synonyms.append(value)
                seen.add(folded)
        self.synonyms = synonyms
        return self


class AtlasOnboardingResult(StrictModel):
    success: Literal[True] = True
    atlas_name: str
    atlas_root: str
    canonical_json: str
    volume: str
    labels: str
    report: str
    canonicalization: Canonicalization
    region_count: int = Field(gt=0)


def _volume_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    raise AtlasOnboardingError(f"Atlas volume must be .nii or .nii.gz: {path}")


def _normalize_text(value: str) -> str:
    value = re.sub(r"[_-]+", " ", value.strip())
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _laterality(value: str) -> Laterality:
    tokens = set(_normalize_text(value).split())
    left = bool(tokens & {"left", "lh"})
    right = bool(tokens & {"right", "rh"})
    if "bilateral" in tokens or left and right:
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    if tokens & {"midline", "median"}:
        return "midline"
    return "unknown"


def _canonical_name(value: str) -> str:
    tokens = _normalize_text(value).split()
    name = " ".join(
        token
        for token in tokens
        if token not in {"left", "right", "lh", "rh", "bilateral"}
    )
    return name or _normalize_text(value)


def _structure_type(canonical_name: str, source_label: str) -> str:
    text = f"{canonical_name} {source_label}".casefold()
    if any(
        term in text
        for term in (
            "abnormality",
            "lesion",
            "infarct",
            "tumor",
            "tumour",
            "edema",
            "haemorrhage",
            "hemorrhage",
        )
    ):
        return "lesion"
    if any(
        term in text
        for term in (
            "artery",
            "arterial",
            "vascular",
            "mca",
            "aca",
            "pca",
            "basilar",
        )
    ):
        return "vascular"
    if "ventricle" in text or "ventricular" in text:
        return "ventricle"
    if "white matter" in text or re.search(r"\bwm\b", text):
        return "white_matter"
    if "brainstem" in text or "brain stem" in text:
        return "brainstem"
    if "cerebell" in text:
        return "cerebellum"
    if any(term in text for term in ("cortex", "gyrus", "sulcus")):
        return "cortical_region"
    if any(
        term in text
        for term in (
            "thalamus",
            "caudate",
            "putamen",
            "pallidum",
            "hippocampus",
            "amygdala",
            "accumbens",
            "diencephalon",
        )
    ):
        return "subcortical_nucleus"
    return "unknown"


def _source_label(
    value: Mapping[str, Any],
    fallback: object,
) -> str:
    for key in ("source_label", "label", "name", "canonical_name"):
        candidate = value.get(key)
        if candidate is not None:
            return str(candidate)
    return str(fallback)


def _load_entries(path: Path) -> list[LabelEntry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AtlasOnboardingError(f"Could not read labels JSON: {path}") from error
    except json.JSONDecodeError as error:
        raise AtlasOnboardingError(f"Labels file is not valid JSON: {path}") from error

    entries = []
    try:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    region_id = int(value.get("region_id", key))
                    entries.append(
                        LabelEntry(
                            region_id,
                            _source_label(value, key),
                            value,
                        )
                    )
                else:
                    entries.append(LabelEntry(int(key), str(value)))
        elif isinstance(payload, list):
            for value in payload:
                if not isinstance(value, dict) or "region_id" not in value:
                    raise AtlasOnboardingError(
                        "List-form labels must be objects with region_id."
                    )
                entries.append(
                    LabelEntry(
                        int(value["region_id"]),
                        _source_label(value, value["region_id"]),
                        value,
                    )
                )
        else:
            raise AtlasOnboardingError(
                "Labels JSON must map IDs to labels or contain region objects."
            )
    except AtlasOnboardingError:
        raise
    except (TypeError, ValueError) as error:
        raise AtlasOnboardingError(
            f"Labels JSON contains an invalid region ID: {error}"
        ) from error

    if not entries:
        raise AtlasOnboardingError("Labels JSON must contain at least one region.")
    invalid_ids = sorted({entry.region_id for entry in entries if entry.region_id <= 0})
    if invalid_ids:
        raise AtlasOnboardingError(
            f"Region IDs must be positive integers: {invalid_ids}"
        )
    ids = [entry.region_id for entry in entries]
    duplicates = sorted(region_id for region_id in set(ids) if ids.count(region_id) > 1)
    if duplicates:
        raise AtlasOnboardingError(f"Duplicate region IDs: {duplicates}")
    return sorted(entries, key=lambda entry: entry.region_id)


def _synonyms(
    canonical_name: str,
    source_label: str,
    laterality: Laterality,
) -> list[str]:
    candidates = [
        canonical_name,
        _normalize_text(source_label),
    ]
    if laterality in {"left", "right"}:
        candidates.append(f"{laterality} {canonical_name}")
    values = []
    seen = set()
    for candidate in candidates:
        folded = candidate.casefold()
        if candidate and folded not in seen:
            values.append(candidate)
            seen.add(folded)
    return values


def _canonicalize_deterministic(entry: LabelEntry) -> CanonicalLabel:
    canonical_name = _canonical_name(entry.source_label)
    laterality = _laterality(entry.source_label)
    structure_type = _structure_type(canonical_name, entry.source_label)
    return CanonicalLabel(
        region_id=entry.region_id,
        source_label=entry.source_label,
        canonical_name=canonical_name,
        laterality=laterality,
        structure_type=structure_type,
        is_abnormality=structure_type == "lesion",
        synonyms=_synonyms(canonical_name, entry.source_label, laterality),
        confidence=0.9,
        rationale=("Deterministic normalization of separators, case, and laterality."),
    )


def _canonicalize_none(entry: LabelEntry) -> CanonicalLabel:
    original = entry.original or {}
    canonical_name = str(
        original.get("canonical_name", _normalize_text(entry.source_label))
    )
    try:
        return CanonicalLabel(
            region_id=entry.region_id,
            source_label=entry.source_label,
            canonical_name=canonical_name,
            parent_anatomy=original.get("parent_anatomy"),
            laterality=original.get("laterality", "unknown"),
            structure_type=str(original.get("structure_type", "unknown")),
            is_abnormality=bool(original.get("is_abnormality", False)),
            synonyms=list(original.get("synonyms") or [canonical_name]),
            confidence=float(original.get("confidence", 1.0)),
            rationale=str(
                original.get(
                    "rationale",
                    "Canonicalization skipped; source metadata retained.",
                )
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise AtlasOnboardingError(
            f"Invalid canonical metadata for region {entry.region_id}: {error}"
        ) from error


def _canonicalize_openai(
    entries: list[LabelEntry],
    *,
    client: OpenAIClient,
    model: str,
    chunk_size: int,
) -> list[CanonicalLabel]:
    regions = []
    source_labels = {entry.region_id: entry.source_label for entry in entries}
    for offset in range(0, len(entries), chunk_size):
        chunk = entries[offset : offset + chunk_size]
        response = request_json(
            client,
            model=model,
            instructions=ATLAS_ONBOARDING_PROMPT,
            payload={
                "labels": [
                    {
                        "region_id": entry.region_id,
                        "source_label": entry.source_label,
                    }
                    for entry in chunk
                ]
            },
        )
        raw_regions = response.get("regions")
        if not isinstance(raw_regions, list):
            raise AtlasOnboardingError(
                "OpenAI canonicalization response must contain a regions list."
            )
        try:
            parsed = [CanonicalLabel.model_validate(region) for region in raw_regions]
        except ValidationError as error:
            raise AtlasOnboardingError(
                f"Invalid OpenAI canonicalization response: {error}"
            ) from error
        expected_ids = {entry.region_id for entry in chunk}
        returned_ids = [region.region_id for region in parsed]
        if set(returned_ids) != expected_ids or len(returned_ids) != len(expected_ids):
            raise AtlasOnboardingError(
                "OpenAI canonicalization must return every input region ID "
                "exactly once."
            )
        regions.extend(
            region.model_copy(update={"source_label": source_labels[region.region_id]})
            for region in parsed
        )
    return sorted(regions, key=lambda region: region.region_id)


def _canonicalize(
    entries: list[LabelEntry],
    *,
    method: Canonicalization,
    client: OpenAIClient | None,
    model: str,
    chunk_size: int,
) -> list[CanonicalLabel]:
    if method == "none":
        return [_canonicalize_none(entry) for entry in entries]
    if method == "deterministic":
        return [_canonicalize_deterministic(entry) for entry in entries]
    if method == "openai":
        if chunk_size < 1:
            raise AtlasOnboardingError("chunk_size must be at least 1.")
        return _canonicalize_openai(
            entries,
            client=client or create_client(),
            model=model,
            chunk_size=chunk_size,
        )
    raise AtlasOnboardingError(f"Unknown canonicalization method: {method}")


def _copy(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def onboard_atlas(
    atlas_name: str,
    atlas_volume: str | Path,
    output_root: str | Path,
    *,
    labels_json: str | Path | None = None,
    canonicalization: Canonicalization = "deterministic",
    model: str = "gpt-6-astra",
    chunk_size: int = 15,
    client: OpenAIClient | None = None,
    overwrite: bool = False,
) -> AtlasOnboardingResult:
    if not atlas_name or Path(atlas_name).name != atlas_name:
        raise AtlasOnboardingError(
            "atlas_name must be a single nonempty directory name."
        )
    volume = Path(atlas_volume).expanduser().resolve()
    if not volume.is_file():
        raise AtlasOnboardingError(f"Atlas volume does not exist: {volume}")
    volume_stem = _volume_stem(volume)
    try:
        data, _ = load_label_volume(volume)
    except (OSError, ValueError) as error:
        raise AtlasOnboardingError(f"Invalid atlas volume: {error}") from error
    atlas_labels = {int(value) for value in np.unique(data) if int(value) != 0}
    if not atlas_labels:
        raise AtlasOnboardingError("Atlas volume has no nonzero labels.")

    generated_labels = labels_json is None
    if generated_labels:
        if atlas_labels != {1}:
            raise AtlasOnboardingError(
                "labels_json is required when an atlas contains more than "
                "one nonzero integer label."
            )
        entries = [LabelEntry(1, volume_stem)]
        labels_source = None
        labels_name = f"{volume_stem}_labels.json"
    else:
        labels_source = Path(labels_json).expanduser().resolve()
        if not labels_source.is_file():
            raise AtlasOnboardingError(f"Labels JSON does not exist: {labels_source}")
        entries = _load_entries(labels_source)
        labels_name = labels_source.name

    entry_ids = {entry.region_id for entry in entries}
    missing_from_volume = sorted(entry_ids - atlas_labels)
    missing_from_labels = sorted(atlas_labels - entry_ids)
    if missing_from_volume or missing_from_labels:
        raise AtlasOnboardingError(
            "Atlas/label mismatch: "
            f"labels absent from volume={missing_from_volume}; "
            f"volume labels absent from JSON={missing_from_labels}."
        )

    regions = _canonicalize(
        entries,
        method=canonicalization,
        client=client,
        model=model,
        chunk_size=chunk_size,
    )
    regions = [
        region.model_copy(
            update={
                "atlas_sources": [volume_stem],
                "preferred_atlas": volume_stem,
            }
        )
        for region in regions
    ]

    atlas_root = Path(output_root).expanduser().resolve() / atlas_name
    raw_labels_dir = atlas_root / "raw_labels"
    volumes_dir = atlas_root / "volumes"
    labels_destination = raw_labels_dir / labels_name
    volume_destination = volumes_dir / volume.name
    canonical_path = atlas_root / f"{volume_stem}_canonical_names.json"
    report_path = atlas_root / f"{volume_stem}_onboarding_report.json"
    targets = [
        labels_destination,
        volume_destination,
        canonical_path,
        report_path,
    ]
    input_for_target = {volume_destination: volume}
    if labels_source is not None:
        input_for_target[labels_destination] = labels_source
    existing = [
        str(path)
        for path in targets
        if path.exists()
        and (
            path not in input_for_target
            or path.resolve() != input_for_target[path].resolve()
        )
    ]
    if existing and not overwrite:
        raise AtlasOnboardingError(
            f"Onboarding outputs already exist; use overwrite=True: {existing}"
        )

    raw_labels_dir.mkdir(parents=True, exist_ok=True)
    volumes_dir.mkdir(parents=True, exist_ok=True)
    if generated_labels:
        labels_destination.write_text(
            json.dumps({"1": volume_stem}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        _copy(labels_source, labels_destination)
    _copy(volume, volume_destination)
    canonical_path.write_text(
        json.dumps(
            [region.model_dump(mode="json") for region in regions],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    result = AtlasOnboardingResult(
        atlas_name=atlas_name,
        atlas_root=str(atlas_root),
        canonical_json=str(canonical_path),
        volume=str(volume_destination),
        labels=str(labels_destination),
        report=str(report_path),
        canonicalization=canonicalization,
        region_count=len(regions),
    )
    report_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Onboard raw labels and a NIfTI volume into a canonical atlas "
            "source directory."
        )
    )
    parser.add_argument("atlas_name")
    parser.add_argument("atlas_volume", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument(
        "--canonicalization",
        choices=["none", "deterministic", "openai"],
        default="deterministic",
    )
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = onboard_atlas(
        args.atlas_name,
        args.atlas_volume,
        args.output_root,
        labels_json=args.labels_json,
        canonicalization=args.canonicalization,
        model=args.model,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
