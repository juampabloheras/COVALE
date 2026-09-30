import json
from collections.abc import Mapping
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from pydantic import ValidationError

from covale.localize.atlas_expression import (
    LocalizationError,
    execute_registry_expression,
)
from covale.localize.spatial_primitives import geometry
from covale.registry import Registry, RegistryError
from covale.registry.models import LabelSource, validate_expression


class RegistryLocalizationTools:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def search_regions(
        self,
        query: str,
        *,
        laterality: str | None = None,
        structure_type: str | None = None,
        atlas: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Search atlas regions for one anatomical concept."""
        matches = self.registry.search(
            query,
            laterality=laterality,
            structure_type=structure_type,
            atlas=atlas,
            limit=min(limit, 100),
        )
        return [
            {
                **match.model_dump(exclude={"region_id"}),
                "id": match.region_id,
                "parent_anatomy": self.registry.model.regions[
                    match.region_id
                ].parent_anatomy,
                "atlas": getattr(
                    self.registry.model.regions[match.region_id].source,
                    "atlas",
                    None,
                ),
            }
            for match in matches
        ]

    def list_regions(
        self,
        *,
        laterality: str | None = None,
        structure_type: str | None = None,
        parent_anatomy: str | None = None,
        atlas: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """List regions by structured ontology fields without fuzzy ranking."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        normalized_parent = parent_anatomy.casefold() if parent_anatomy else None
        entries = []
        for region_id, region in self.registry.model.regions.items():
            source_atlas = getattr(region.source, "atlas", None)
            if laterality and region.laterality != laterality:
                continue
            if structure_type and region.structure_type != structure_type:
                continue
            if atlas and source_atlas != atlas:
                continue
            if normalized_parent and (
                region.parent_anatomy is None
                or normalized_parent not in region.parent_anatomy.casefold()
            ):
                continue
            entries.append(self._region_entry(region_id))
        entries.sort(
            key=lambda entry: (
                str(entry["laterality"]),
                str(entry["name"]),
                str(entry["id"]),
            )
        )
        return entries[: min(limit, 250)]

    def get_region(self, region_id: str) -> dict[str, object]:
        """Return complete ontology metadata for one stable region ID."""
        if region_id not in self.registry.model.regions:
            raise ValueError(f"Unknown registry region ID: {region_id}")
        return self._region_entry(region_id)

    def validate_expression(
        self,
        expression: Mapping[str, Any],
    ) -> dict[str, object]:
        """Validate and execute a proposed atlas expression."""
        try:
            normalized = validate_expression(expression).model_dump(
                exclude_none=True
            )
            if _component_count(normalized) > 20:
                raise ValueError(
                    "Atlas expressions may contain at most 20 region components."
                )
            mask = execute_registry_expression(normalized, self.registry)
        except (ValidationError, LocalizationError, RegistryError, ValueError) as error:
            return {
                "valid": False,
                "error": str(error),
            }
        return {
            "valid": True,
            "expression": normalized,
            "component_ids": sorted(_region_ids(normalized)),
            "voxel_count": int(np.count_nonzero(mask.data)),
            "shape": list(mask.data.shape),
        }

    def inspect_expression_geometry(
        self,
        expression: Mapping[str, Any],
    ) -> dict[str, object]:
        """Return physical RAS geometry for an executable expression."""
        try:
            normalized = validate_expression(expression).model_dump(
                exclude_none=True
            )
            mask = execute_registry_expression(normalized, self.registry)
            return {
                "valid": True,
                "expression": normalized,
                **geometry(mask),
            }
        except (ValidationError, LocalizationError, RegistryError, ValueError) as error:
            return {
                "valid": False,
                "error": str(error),
            }

    def tool_functions(self) -> list[object]:
        """Return callables suitable for Deep Agent tool registration."""

        def search_regions(
            query: str,
            laterality: str | None = None,
            structure_type: str | None = None,
            atlas: str | None = None,
            limit: int = 20,
        ) -> list[dict[str, object]]:
            """Search the atlas registry for one anatomical concept."""
            return self.search_regions(
                query,
                laterality=laterality,
                structure_type=structure_type,
                atlas=atlas,
                limit=limit,
            )

        def list_regions(
            laterality: str | None = None,
            structure_type: str | None = None,
            parent_anatomy: str | None = None,
            atlas: str | None = None,
            limit: int = 100,
        ) -> list[dict[str, object]]:
            """Browse atlas regions using structured ontology filters."""
            return self.list_regions(
                laterality=laterality,
                structure_type=structure_type,
                parent_anatomy=parent_anatomy,
                atlas=atlas,
                limit=limit,
            )

        def get_region(region_id: str) -> dict[str, object]:
            """Get complete metadata for one stable atlas region ID."""
            return self.get_region(region_id)

        def validate_atlas_expression(
            expression: dict[str, Any],
        ) -> dict[str, object]:
            """Validate and execute an atlas expression before returning it."""
            return self.validate_expression(expression)

        def inspect_expression_geometry(
            expression: dict[str, Any],
        ) -> dict[str, object]:
            """Inspect mask bounds and centroid in RAS millimeters."""
            return self.inspect_expression_geometry(expression)

        return [
            search_regions,
            list_regions,
            get_region,
            inspect_expression_geometry,
            validate_atlas_expression,
        ]

    def _region_entry(self, region_id: str) -> dict[str, object]:
        region = self.registry.model.regions[region_id]
        source = region.source
        return {
            "id": region_id,
            "name": region.display_name,
            "canonical_name": region.canonical_name,
            "source_label": region.source_label,
            "laterality": region.laterality,
            "parent_anatomy": region.parent_anatomy,
            "structure_type": region.structure_type,
            "synonyms": region.synonyms,
            "atlas": source.atlas if isinstance(source, LabelSource) else None,
        }


def tavily_search(
    query: str,
    *,
    api_key: str,
    max_results: int = 5,
) -> dict[str, object]:
    """Search external anatomy references through the Tavily API."""
    if not api_key:
        raise ValueError("TAVILY_API_KEY is required for web search.")
    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": min(max(max_results, 1), 10),
            "include_answer": False,
        }
    ).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Tavily response must be a JSON object.")
    return {
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
            }
            for item in result.get("results", [])
            if isinstance(item, dict)
        ]
    }


def _region_ids(expression: Mapping[str, Any]) -> set[str]:
    if expression["op"] == "unresolved":
        return set()
    if expression["op"] == "region":
        region_id = expression.get("id")
        return {region_id} if isinstance(region_id, str) else set()
    if "arg" in expression:
        return _region_ids(expression["arg"])
    return {
        region_id
        for argument in expression["args"]
        for region_id in _region_ids(argument)
    }


def _component_count(expression: Mapping[str, Any]) -> int:
    if expression["op"] == "unresolved":
        return 0
    if expression["op"] == "region":
        return 1
    if "arg" in expression:
        return _component_count(expression["arg"])
    return sum(_component_count(argument) for argument in expression["args"])
