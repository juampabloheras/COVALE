import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from covale.registry.models import (
    AtlasSpace,
    LegacyAtlasRegistry,
    MaskSource,
    RegionDefinition,
    RegionMatch,
    RegistryV2,
)


class RegistryError(ValueError):
    pass


def normalize_term(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    tokens = [
        "left" if token in {"l", "lh"} else "right" if token in {"r", "rh"} else token
        for token in value.split()
    ]
    return " ".join(tokens)


def query_laterality(value: str) -> str | None:
    tokens = set(normalize_term(value).split())
    has_left = "left" in tokens
    has_right = "right" in tokens
    if has_left and has_right or "bilateral" in tokens:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    if "midline" in tokens or "median" in tokens:
        return "midline"
    return None


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", normalize_term(value))
    return value.strip("-") or "region"


def _convert_v1(registry: LegacyAtlasRegistry) -> RegistryV2:
    regions = {}
    used_ids: set[str] = set()
    for name, region in registry.regions.items():
        base_id = f"legacy:{_slug(name)}"
        region_id = base_id
        suffix = 2
        while region_id in used_ids:
            region_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(region_id)
        regions[region_id] = RegionDefinition(
            canonical_name=name,
            display_name=name,
            synonyms=region.aliases,
            source=MaskSource(type="mask", path=region.mask),
        )
    if not regions:
        raise RegistryError("Registry must define at least one region.")
    return RegistryV2(
        schema_version=2,
        space=AtlasSpace(name=registry.space),
        atlases={},
        regions=regions,
    )


class Registry:
    def __init__(self, model: RegistryV2, root: Path) -> None:
        self.model = model
        self.root = root.resolve()
        self._atlas_cache: dict[str, Any] = {}
        self._mask_cache: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise RegistryError("Registry JSON must be an object.")
            if payload.get("schema_version") == 2:
                model = RegistryV2.model_validate(payload)
            else:
                model = _convert_v1(LegacyAtlasRegistry.model_validate(payload))
        except OSError as error:
            raise RegistryError(f"Could not read registry: {path}") from error
        except json.JSONDecodeError as error:
            raise RegistryError(f"Registry is not valid JSON: {path}") from error
        except ValidationError as error:
            raise RegistryError(f"Invalid registry {path}: {error}") from error
        return cls(model, path.parent)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        laterality: str | None = None,
        structure_type: str | None = None,
        atlas: str | None = None,
        parent_id: str | None = None,
        include_abnormalities: bool = False,
    ) -> list[RegionMatch]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")
        normalized_query = normalize_term(query)
        selected_laterality = laterality or query_laterality(query)
        matches = []

        for region_id, region in self.model.regions.items():
            if region.is_abnormality and not include_abnormalities:
                continue
            if selected_laterality:
                allowed_lateralities = {
                    selected_laterality,
                    "unknown",
                }
                if selected_laterality == "bilateral":
                    allowed_lateralities.update({"left", "right"})
                if region.laterality not in allowed_lateralities:
                    continue
            if structure_type and region.structure_type != structure_type:
                continue
            if parent_id and parent_id not in region.parent_ids:
                continue
            if atlas:
                source_atlas = getattr(region.source, "atlas", None)
                if source_atlas != atlas:
                    continue

            terms = [
                ("id", region_id),
                ("display_name", region.display_name),
                ("canonical_name", region.canonical_name),
            ]
            if region.source_label:
                terms.append(("source_label", region.source_label))
            terms.extend(("synonym", synonym) for synonym in region.synonyms)

            exact = [
                (match_type, term)
                for match_type, term in terms
                if normalize_term(term) == normalized_query
            ]
            if exact:
                match_type, matched_term = exact[0]
                score = 1.0
            else:
                scored = [
                    (
                        SequenceMatcher(
                            None,
                            normalized_query,
                            normalize_term(term),
                        ).ratio(),
                        term,
                    )
                    for _, term in terms
                ]
                score, matched_term = max(scored)
                match_type = "approximate"

            matches.append(
                RegionMatch(
                    region_id=region_id,
                    display_name=region.display_name,
                    matched_term=matched_term,
                    match_type=match_type,
                    score=score,
                    laterality=region.laterality,
                    structure_type=region.structure_type,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                self._priority(match.region_id),
                match.region_id,
            )
        )
        return matches[:limit]

    def compact_catalog(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return [
            {
                "id": match.region_id,
                "name": match.display_name,
                "laterality": match.laterality,
                "structure_type": match.structure_type,
                "synonyms": self.model.regions[match.region_id].synonyms,
            }
            for match in self.search(query, limit=limit)
        ]

    def resolve_name(self, name: str) -> str | None:
        exact = [
            match
            for match in self.search(name, limit=len(self.model.regions))
            if match.score == 1.0
        ]
        if not exact:
            return None
        region_ids = {match.region_id for match in exact}
        if len(region_ids) != 1:
            raise RegistryError(
                f"Ambiguous atlas region '{name}': {sorted(region_ids)}"
            )
        return exact[0].region_id

    def mask(self, region_id: str):
        from covale.registry.masks import load_region_mask

        if region_id not in self._mask_cache:
            self._mask_cache[region_id] = load_region_mask(
                self.model,
                self.root,
                region_id,
                atlas_cache=self._atlas_cache,
            )
        return self._mask_cache[region_id]

    def _priority(self, region_id: str) -> int:
        source = self.model.regions[region_id].source
        atlas_id = getattr(source, "atlas", None)
        return self.model.atlases[atlas_id].priority if atlas_id is not None else 0
