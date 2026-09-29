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
from covale.prompts import ATLAS_ONBOARDING_PROMPT
from covale.registry import Registry
from covale.registry.masks import load_label_volume, sha256_file
from covale.registry.models import (
    AtlasDefinition,
    LabelSource,
    Laterality,
    RegionDefinition,
    RegistryV2,
)
from covale.utils.models import StrictModel

Canonicalization = Literal["none", "deterministic", "openai"]


class AtlasOnboardingError(ValueError):
    pass


@dataclass(frozen=True)
class LabelEntry:
    region_id: int
    source_label: str
    stable_id: str | None = None
    original: Mapping[str, Any] | None = None


class AtlasMetadata(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=1)
    role: str = Field(default="anatomical", min_length=1)
    priority: int = Field(default=100, ge=0)
    source_url: str = Field(min_length=1)
    version: str | None = None
    citation: str = Field(min_length=1)
    license: str = Field(min_length=1)


class CanonicalLabel(StrictModel):
    id: str | None = Field(default=None, min_length=1)
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
    metadata: str
    license_file: str
    report: str
    registry: str
    atlas_id: str
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


def _stable_id(value: Mapping[str, Any]) -> str | None:
    stable_id = value.get("id", value.get("stable_id"))
    return str(stable_id) if stable_id is not None else None


def _load_entries(path: Path, atlas_id: str) -> list[LabelEntry]:
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
                            _stable_id(value) or f"{atlas_id}:{region_id}",
                            value,
                        )
                    )
                else:
                    region_id = int(key)
                    entries.append(
                        LabelEntry(
                            region_id,
                            str(value),
                            f"{atlas_id}:{region_id}",
                        )
                    )
        elif isinstance(payload, list):
            for value in payload:
                if not isinstance(value, dict) or "region_id" not in value:
                    raise AtlasOnboardingError(
                        "List-form labels must be objects with region_id."
                    )
                entries.append(
                    LabelEntry(
                        region_id := int(value["region_id"]),
                        _source_label(value, value["region_id"]),
                        _stable_id(value) or f"{atlas_id}:{region_id}",
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
    stable_ids = [entry.stable_id for entry in entries if entry.stable_id is not None]
    duplicate_stable_ids = sorted(
        stable_id for stable_id in set(stable_ids) if stable_ids.count(stable_id) > 1
    )
    if duplicate_stable_ids:
        raise AtlasOnboardingError(
            f"Duplicate stable region IDs: {duplicate_stable_ids}"
        )
    invalid_stable_ids = [
        entry.stable_id
        for entry in entries
        if not entry.stable_id.startswith(f"{atlas_id}:")
    ]
    if invalid_stable_ids:
        raise AtlasOnboardingError(
            f"Stable region IDs must start with '{atlas_id}:': " f"{invalid_stable_ids}"
        )
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
        id=entry.stable_id,
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
            id=entry.stable_id,
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
    source_entries = {entry.region_id: entry for entry in entries}
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
            region.model_copy(
                update={
                    "id": source_entries[region.region_id].stable_id,
                    "source_label": source_entries[region.region_id].source_label,
                }
            )
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


def _display_name(region: CanonicalLabel) -> str:
    if region.laterality in {"left", "right", "bilateral"}:
        prefix = f"{region.laterality} "
        if not region.canonical_name.casefold().startswith(prefix):
            return f"{prefix}{region.canonical_name}"
    return region.canonical_name


def _prepare_registry_update(
    registry_path: Path,
    metadata: AtlasMetadata,
    regions: list[CanonicalLabel],
    volume_source: Path,
    volume_destination: Path,
    *,
    overwrite: bool,
) -> RegistryV2:
    if not registry_path.is_file():
        raise AtlasOnboardingError(f"Atlas registry does not exist: {registry_path}")
    registry = Registry.load(registry_path)
    try:
        relative_volume = volume_destination.relative_to(
            registry_path.parent.resolve()
        ).as_posix()
    except ValueError as error:
        raise AtlasOnboardingError(
            "Onboarded atlas assets must remain inside the registry directory."
        ) from error

    atlases = dict(registry.model.atlases)
    registry_regions = dict(registry.model.regions)
    if metadata.id in atlases:
        if not overwrite:
            raise AtlasOnboardingError(
                f"Atlas ID already exists in registry: {metadata.id}"
            )
        registry_regions = {
            region_id: definition
            for region_id, definition in registry_regions.items()
            if getattr(definition.source, "atlas", None) != metadata.id
        }

    stable_ids = [region.id for region in regions]
    if any(stable_id is None for stable_id in stable_ids):
        raise AtlasOnboardingError(
            "Registry updates require a stable id for every region."
        )
    conflicts = sorted(
        stable_id for stable_id in stable_ids if stable_id in registry_regions
    )
    if conflicts:
        raise AtlasOnboardingError(
            f"Stable region IDs already exist in registry: {conflicts}"
        )

    atlases[metadata.id] = AtlasDefinition(
        name=metadata.name,
        volume=relative_volume,
        role=metadata.role,
        priority=metadata.priority,
        source_url=metadata.source_url,
        version=metadata.version,
        citation=metadata.citation,
        license=metadata.license,
        sha256=sha256_file(volume_source),
    )
    for region in regions:
        if region.id is None:
            raise AtlasOnboardingError(
                f"Region label {region.region_id} has no stable id."
            )
        registry_regions[region.id] = RegionDefinition(
            canonical_name=region.canonical_name,
            display_name=_display_name(region),
            source_label=region.source_label,
            laterality=region.laterality,
            parent_anatomy=region.parent_anatomy,
            structure_type=region.structure_type,
            is_abnormality=region.is_abnormality,
            synonyms=region.synonyms,
            confidence=region.confidence,
            rationale=region.rationale,
            source=LabelSource(
                type="labels",
                atlas=metadata.id,
                values=[region.region_id],
            ),
        )
    try:
        return RegistryV2(
            schema_version=2,
            space=registry.model.space,
            atlases=atlases,
            regions=registry_regions,
        )
    except ValidationError as error:
        raise AtlasOnboardingError(
            f"Onboarding would create an invalid registry: {error}"
        ) from error


def _write_registry(path: Path, registry: RegistryV2) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(registry.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_metadata(path: Path) -> AtlasMetadata:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AtlasMetadata.model_validate(payload)
    except OSError as error:
        raise AtlasOnboardingError(f"Could not read atlas metadata: {path}") from error
    except json.JSONDecodeError as error:
        raise AtlasOnboardingError(
            f"Atlas metadata is not valid JSON: {path}"
        ) from error
    except ValidationError as error:
        raise AtlasOnboardingError(f"Invalid atlas metadata {path}: {error}") from error


def _discover_volume(atlas_dir: Path) -> Path:
    volumes = sorted(atlas_dir.glob("*.nii")) + sorted(atlas_dir.glob("*.nii.gz"))
    if len(volumes) != 1:
        raise AtlasOnboardingError(
            "Atlas directory must contain exactly one .nii or .nii.gz "
            f"volume; found {len(volumes)}."
        )
    return volumes[0].resolve()


def onboard_atlas(
    atlas_dir: str | Path,
    *,
    canonicalization: Canonicalization = "deterministic",
    model: str = "gpt-6-astra",
    chunk_size: int = 15,
    client: OpenAIClient | None = None,
    registry_path: str | Path = "atlas_registry/registry.json",
    overwrite: bool = False,
) -> AtlasOnboardingResult:
    source_dir = Path(atlas_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise AtlasOnboardingError(f"Atlas directory does not exist: {source_dir}")
    atlas_name = source_dir.name
    if not atlas_name:
        raise AtlasOnboardingError("Atlas directory must have a name.")

    metadata_source = source_dir / "atlas.json"
    labels_source = source_dir / "labels.json"
    license_source = source_dir / "LICENSE.txt"
    missing_files = [
        path.name
        for path in (metadata_source, labels_source, license_source)
        if not path.is_file()
    ]
    if missing_files:
        raise AtlasOnboardingError(
            f"Atlas directory is missing required files: {missing_files}"
        )
    metadata = _load_metadata(metadata_source)
    if not license_source.read_text(encoding="utf-8").strip():
        raise AtlasOnboardingError("LICENSE.txt must not be empty.")

    volume = _discover_volume(source_dir)
    volume_stem = _volume_stem(volume)
    try:
        data, _ = load_label_volume(volume)
    except (OSError, ValueError) as error:
        raise AtlasOnboardingError(f"Invalid atlas volume: {error}") from error
    atlas_labels = {int(value) for value in np.unique(data) if int(value) != 0}
    if not atlas_labels:
        raise AtlasOnboardingError("Atlas volume has no nonzero labels.")

    entries = _load_entries(labels_source, metadata.id)

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

    resolved_registry_path = Path(registry_path).expanduser().resolve()
    atlas_root = resolved_registry_path.parent / "atlases" / atlas_name
    raw_labels_dir = atlas_root / "raw_labels"
    volumes_dir = atlas_root / "volumes"
    labels_destination = raw_labels_dir / labels_source.name
    metadata_destination = atlas_root / metadata_source.name
    license_destination = atlas_root / license_source.name
    volume_destination = volumes_dir / volume.name
    canonical_path = atlas_root / f"{volume_stem}_canonical_names.json"
    report_path = atlas_root / f"{volume_stem}_onboarding_report.json"
    targets = [
        labels_destination,
        metadata_destination,
        license_destination,
        volume_destination,
        canonical_path,
        report_path,
    ]
    input_for_target = {
        labels_destination: labels_source,
        metadata_destination: metadata_source,
        license_destination: license_source,
        volume_destination: volume,
    }
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

    updated_registry = _prepare_registry_update(
        resolved_registry_path,
        metadata,
        regions,
        volume,
        volume_destination,
        overwrite=overwrite,
    )

    raw_labels_dir.mkdir(parents=True, exist_ok=True)
    volumes_dir.mkdir(parents=True, exist_ok=True)
    _copy(labels_source, labels_destination)
    _copy(metadata_source, metadata_destination)
    _copy(license_source, license_destination)
    _copy(volume, volume_destination)
    canonical_path.write_text(
        json.dumps(
            [region.model_dump(mode="json") for region in regions],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_registry(resolved_registry_path, updated_registry)

    result = AtlasOnboardingResult(
        atlas_name=atlas_name,
        atlas_root=str(atlas_root),
        canonical_json=str(canonical_path),
        volume=str(volume_destination),
        labels=str(labels_destination),
        metadata=str(metadata_destination),
        license_file=str(license_destination),
        report=str(report_path),
        registry=str(resolved_registry_path),
        atlas_id=metadata.id,
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
        description=("Onboard an atlas directory and update the COVALE registry.")
    )
    parser.add_argument("atlas_dir", type=Path)
    parser.add_argument(
        "--canonicalization",
        choices=["none", "deterministic", "openai"],
        default="deterministic",
    )
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("atlas_registry/registry.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = onboard_atlas(
        args.atlas_dir,
        canonicalization=args.canonicalization,
        model=args.model,
        chunk_size=args.chunk_size,
        registry_path=args.registry,
        overwrite=args.overwrite,
    )
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
