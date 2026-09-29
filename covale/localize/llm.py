from collections.abc import Mapping
from typing import Any

from covale.localize.atlas_expression import (
    Expression,
    LOCALIZATION_PROMPT,
    LocalizationError,
)
from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)


def build_expression(
    text: str,
    registry: Mapping[str, Any],
    *,
    client: OpenAIClient | None = None,
    model: str = "gpt-6-astra",
) -> Expression:
    client = client or create_client()
    expression = request_json(
        client,
        model=model,
        instructions=LOCALIZATION_PROMPT,
        payload={
            "anatomical_description": text,
            "atlas_registry": dict(registry),
        },
    )
    if not isinstance(expression.get("op"), str):
        raise LocalizationError("LLM localization response must contain a string op.")
    return expression
