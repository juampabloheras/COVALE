from collections.abc import Mapping
from typing import Any

from covale.localize.atlas_expression import Expression, LocalizationError
from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
)
from covale.registry.resolve import RegistryResolver


def build_expression(
    text: str,
    registry: Mapping[str, Any],
    *,
    client: OpenAIClient | None = None,
    model: str = "gpt-6-astra",
) -> Expression:
    client = client or create_client()
    expression = RegistryResolver.from_mapping(
        registry,
        client=client,
        model=model,
    ).resolve(text)
    if not isinstance(expression.get("op"), str):
        raise LocalizationError("LLM localization response must contain a string op.")
    return expression
