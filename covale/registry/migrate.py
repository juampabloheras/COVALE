import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, ValidationError, model_validator

from covale.models import StrictModel
from covale.registry.masks import load_label_volume, sha256_file
from covale.registry.models import (
    AtlasDefinition,
    AtlasSpace,
    LabelSource,
    Laterality,
    RegionDefinition,
    RegistryV2,
)


class MigrationError(ValueError):
    pass


class CanonicalRegion(StrictModel):
    region_id: int = Field(gt=0)
    source_label: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    parent_anatomy: str | None = None
    laterality: Laterality = "unknown"
    structure_type: str = Field(default="unknown", min_length=1)
    is_abnormality: bool = False
    synonyms: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    atlas_sources: list[str] = Field(default_factory=list)
    preferred_atlas: str | None = None

    @model_validator(mode="after")
    def require_atlas_source(self) -> "CanonicalRegion":
        if self.preferred_atlas and self.preferred_atlas not in self.atlas_sources:
            self.atlas_sources.insert(0, self.preferred_atlas)
        if not self.atlas_sources:
            raise ValueError("Canonical region must identify an atlas source.")
        return self


class AtlasMigration(StrictModel):
    id: str
    source: str
    output: str
    region_count: int = Field(ge=0)
    sha256: str
    missing_license: bool
    missing_citation: bool


class MigrationReport(StrictModel):
    schema_version: Literal[1] = 1
    source_root: str
    output_registry: str
    canonical_files: int = Field(ge=0)
    atlas_count: int = Field(ge=0)
    region_count: int = Field(ge=0)
    atlases: list[AtlasMigration]
    warnings: list[str] = Field(default_factory=list)


def _atlas_name(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise MigrationError(f"Could not create stable ID from: {value!r}")
    return slug


def _display_name(region: CanonicalRegion) -> str:
    canonical = region.canonical_name.strip()
    if region.laterality in {"left", "right", "bilateral"}:
        prefix = f"{region.laterality} "
        if not canonical.casefold().startswith(prefix):
            return f"{prefix}{canonical}"
    return canonical


def _read_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MigrationError(f"Could not read atlas metadata: {path}") from error
    except json.JSONDecodeError as error:
        raise MigrationError(f"Atlas metadata is not valid JSON: {path}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, dict)
        for key, value in payload.items()
    ):
        raise MigrationError(
            "Atlas metadata must be an object mapping atlas names to objects."
        )
    return payload


def _discover_volumes(source_root: Path) -> dict[str, Path]:
    volumes: dict[str, Path] = {}
    for path in sorted(source_root.rglob("*.nii")) + sorted(
        source_root.rglob("*.nii.gz")
    ):
        name = _atlas_name(path)
        if name in volumes and volumes[name] != path:
            raise MigrationError(
                f"Duplicate atlas volume name '{name}': " f"{volumes[name]} and {path}"
            )
        volumes[name] = path
    if not volumes:
        raise MigrationError(f"No NIfTI atlas volumes found under {source_root}.")
    return volumes


def _read_regions(path: Path) -> list[CanonicalRegion]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise MigrationError(f"Canonical ontology must be a JSON array: {path}")
        return [CanonicalRegion.model_validate(item) for item in payload]
    except OSError as error:
        raise MigrationError(f"Could not read canonical ontology: {path}") from error
    except json.JSONDecodeError as error:
        raise MigrationError(f"Canonical ontology is not valid JSON: {path}") from error
    except ValidationError as error:
        raise MigrationError(f"Invalid canonical ontology {path}: {error}") from error


def migrate_registry(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    space: str,
    resolution: str | None = None,
    metadata_path: str | Path | None = None,
    overwrite: bool = False,
) -> MigrationReport:
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    registry_path = output_dir / "registry.json"
    report_path = output_dir / "migration_report.json"
    if not source_root.is_dir():
        raise MigrationError(f"Atlas source directory does not exist: {source_root}")
    if not space.strip():
        raise MigrationError("Atlas space must not be empty.")
    if not overwrite and (registry_path.exists() or report_path.exists()):
        raise MigrationError(
            f"Migration output already exists in {output_dir}; use overwrite=True."
        )

    canonical_paths = sorted(source_root.rglob("*_canonical_names.json"))
    if not canonical_paths:
        raise MigrationError(
            f"No *_canonical_names.json files found under {source_root}."
        )
    volumes = _discover_volumes(source_root)
    metadata = _read_metadata(
        Path(metadata_path).resolve() if metadata_path is not None else None
    )
    unknown_metadata = sorted(set(metadata) - set(volumes))
    if unknown_metadata:
        raise MigrationError(
            f"Metadata references unknown atlas volumes: {unknown_metadata}"
        )
    label_sets: dict[str, set[int]] = {}
    atlas_ids: dict[str, str] = {}
    atlas_models: dict[str, AtlasDefinition] = {}
    atlas_paths: dict[str, Path] = {}

    for name, path in volumes.items():
        atlas_id = _slug(name)
        if atlas_id in atlas_models:
            raise MigrationError(f"Duplicate stable atlas ID: {atlas_id}")
        data, _ = load_label_volume(path)
        label_sets[name] = {int(value) for value in np.unique(data)}
        atlas_ids[name] = atlas_id
        suffix = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"
        relative_output = f"atlases/{atlas_id}{suffix}"
        fields = metadata.get(name, {})
        try:
            atlas_models[atlas_id] = AtlasDefinition(
                name=str(fields.get("name", name)),
                volume=relative_output,
                role=str(fields.get("role", "anatomical")),
                priority=int(fields.get("priority", 100)),
                source_url=fields.get("source_url"),
                version=fields.get("version"),
                citation=fields.get("citation"),
                license=fields.get("license"),
                sha256=sha256_file(path),
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MigrationError(
                f"Invalid metadata for atlas '{name}': {error}"
            ) from error
        atlas_paths[atlas_id] = path

    regions: dict[str, RegionDefinition] = {}
    atlas_region_counts = dict.fromkeys(atlas_models, 0)
    for canonical_path in canonical_paths:
        for region in _read_regions(canonical_path):
            for atlas_name in region.atlas_sources:
                if atlas_name not in atlas_ids:
                    raise MigrationError(
                        f"Region {region.region_id} in {canonical_path} "
                        f"references missing atlas '{atlas_name}'."
                    )
                if region.region_id not in label_sets[atlas_name]:
                    raise MigrationError(
                        f"Region {region.region_id} in {canonical_path} is "
                        f"missing from atlas '{atlas_name}'."
                    )
                atlas_id = atlas_ids[atlas_name]
                stable_id = f"{atlas_id}:{region.region_id}"
                definition = RegionDefinition(
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
                        atlas=atlas_id,
                        values=[region.region_id],
                    ),
                )
                existing = regions.get(stable_id)
                if existing is not None and existing != definition:
                    raise MigrationError(
                        f"Conflicting canonical definitions for '{stable_id}'."
                    )
                if existing is None:
                    regions[stable_id] = definition
                    atlas_region_counts[atlas_id] += 1

    try:
        registry = RegistryV2(
            schema_version=2,
            space=AtlasSpace(name=space, resolution=resolution),
            atlases=atlas_models,
            regions=regions,
        )
    except ValidationError as error:
        raise MigrationError(f"Migrated registry is invalid: {error}") from error

    warnings = []
    migrations = []
    for atlas_id, atlas in registry.atlases.items():
        if atlas.license is None:
            warnings.append(f"Atlas '{atlas_id}' has no license metadata.")
        if atlas.citation is None:
            warnings.append(f"Atlas '{atlas_id}' has no citation metadata.")
        source = atlas_paths[atlas_id]
        migrations.append(
            AtlasMigration(
                id=atlas_id,
                source=str(source),
                output=atlas.volume,
                region_count=atlas_region_counts[atlas_id],
                sha256=atlas.sha256 or "",
                missing_license=atlas.license is None,
                missing_citation=atlas.citation is None,
            )
        )

    report = MigrationReport(
        source_root=str(source_root),
        output_registry=str(registry_path),
        canonical_files=len(canonical_paths),
        atlas_count=len(registry.atlases),
        region_count=len(registry.regions),
        atlases=migrations,
        warnings=warnings,
    )

    destinations = {
        atlas_id: output_dir / registry.atlases[atlas_id].volume
        for atlas_id in atlas_paths
    }
    existing_assets = [str(path) for path in destinations.values() if path.exists()]
    if existing_assets and not overwrite:
        raise MigrationError(
            f"Migration assets already exist; use overwrite=True: {existing_assets}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atlases").mkdir(exist_ok=True)
    for atlas_id, source in atlas_paths.items():
        shutil.copy2(source, destinations[atlas_id])
    registry_path.write_text(
        json.dumps(registry.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MedPrimitives-style atlas assets to COVALE Registry v2."
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--space", required=True)
    parser.add_argument("--resolution")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    migrate_registry(
        args.source_root,
        args.output_dir,
        space=args.space,
        resolution=args.resolution,
        metadata_path=args.metadata,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
