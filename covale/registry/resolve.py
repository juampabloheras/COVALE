from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from covale.localize.openai_client import (
    ModelResponseError,
    OpenAIClient,
    request_json,
)
from covale.prompts import LOCALIZATION_PROMPT
from covale.registry.load import Registry, RegistryError
from covale.registry.models import RegistryV2, validate_expression


class RegistryResolver:
    def __init__(
        self,
        registry: Registry,
        *,
        client: OpenAIClient,
        model: str,
        candidate_limit: int = 20,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1.")
        self.registry = registry
        self.client = client
        self.model = model
        self.candidate_limit = candidate_limit

    @classmethod
    def from_mapping(
        cls,
        registry: Mapping[str, Any],
        *,
        client: OpenAIClient,
        model: str,
        candidate_limit: int = 20,
    ) -> "RegistryResolver":
        try:
            model_registry = RegistryV2.model_validate(registry)
        except ValidationError as error:
            raise ModelResponseError(
                f"Invalid atlas registry supplied to resolver: {error}"
            ) from error
        return cls(
            Registry(model_registry, Path.cwd()),
            client=client,
            model=model,
            candidate_limit=candidate_limit,
        )

    def resolve(self, query: str) -> dict[str, Any]:
        matches = self.registry.search(
            query,
            limit=self.candidate_limit,
        )
        exact_ids = {match.region_id for match in matches if match.score == 1.0}
        if len(exact_ids) == 1:
            return {"op": "region", "id": exact_ids.pop()}

        catalog = self.registry.compact_catalog(
            query,
            limit=self.candidate_limit,
        )
        allowed_ids = {entry["id"] for entry in catalog}
        response = request_json(
            self.client,
            model=self.model,
            instructions=LOCALIZATION_PROMPT,
            payload={
                "anatomical_description": query,
                "candidate_regions": catalog,
            },
        )
        return self._validated_expression(response, allowed_ids)

    def _validated_expression(
        self,
        response: object,
        allowed_ids: set[str],
    ) -> dict[str, Any]:
        try:
            expression = validate_expression(response).model_dump(exclude_none=True)
        except ValidationError as error:
            raise ModelResponseError(
                f"Invalid localization expression: {error}"
            ) from error

        def normalize(node: dict[str, Any]) -> dict[str, Any]:
            if node["op"] == "unresolved":
                return node
            if node["op"] == "region":
                region_id = node.get("id")
                if region_id is None:
                    try:
                        region_id = self.registry.resolve_name(node["name"])
                    except RegistryError as error:
                        raise ModelResponseError(str(error)) from error
                    if region_id is None:
                        raise ModelResponseError(
                            f"Unknown atlas region: {node['name']}"
                        )
                if region_id not in allowed_ids:
                    raise ModelResponseError(
                        f"Localization returned region outside candidate "
                        f"catalog: {region_id}"
                    )
                return {"op": "region", "id": region_id}
            return {
                "op": node["op"],
                "args": [normalize(argument) for argument in node["args"]],
            }

        return normalize(expression)
