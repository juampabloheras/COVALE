import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from covale.localize.atlas_expression import (
    Expression,
    LOCALIZATION_PROMPT,
    LocalizationError,
)
from covale.localize.openai_client import ProviderError


class DeepAgent(Protocol):
    def invoke(self, request: dict[str, object]) -> Mapping[str, Any]: ...


def create_agent(model: str = "gpt-6-astra") -> DeepAgent:
    try:
        from deepagents import create_deep_agent
        from dotenv import load_dotenv
        from openai import OpenAIError
    except ImportError as error:
        raise ProviderError(
            "Deep Agents is a core COVALE dependency. Reinstall COVALE to "
            "restore the provider."
        ) from error

    load_dotenv()
    provider_model = model if ":" in model else f"openai:{model}"
    try:
        return create_deep_agent(
            model=provider_model,
            tools=[],
            system_prompt=LOCALIZATION_PROMPT,
        )
    except (OpenAIError, RuntimeError) as error:
        raise ProviderError(f"Could not initialize Deep Agent: {error}") from error


def read_agent_response(result: Mapping[str, Any]) -> Expression:
    structured_response = result.get("structured_response")
    if isinstance(structured_response, Mapping):
        return dict(structured_response)

    messages = result.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise LocalizationError("Deep agent result did not contain messages.")

    message = messages[-1]
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping)
            and isinstance(block.get("text"), str)
        )
    if not isinstance(content, str):
        raise LocalizationError("Deep agent response did not contain text.")

    try:
        expression = json.loads(content)
    except json.JSONDecodeError as error:
        raise LocalizationError(
            "Deep agent response was not valid JSON."
        ) from error
    if not isinstance(expression, dict) or not isinstance(
        expression.get("op"), str
    ):
        raise LocalizationError(
            "Deep agent response must contain a JSON expression with an op."
        )
    return expression


def build_expression(
    text: str,
    registry: Mapping[str, Any],
    *,
    agent: DeepAgent | None = None,
    model: str = "gpt-6-astra",
) -> Expression:
    agent = agent or create_agent(model)
    try:
        from openai import OpenAIError
    except ImportError as error:
        raise ProviderError(
            "Deep Agents is a core COVALE dependency. Reinstall COVALE to "
            "restore the provider."
        ) from error

    try:
        result = agent.invoke(
            {
                "messages": json.dumps(
                    {
                        "anatomical_description": text,
                        "atlas_registry": dict(registry),
                    }
                )
            }
        )
    except (OpenAIError, RuntimeError) as error:
        raise ProviderError(f"Deep Agent request failed: {error}") from error
    return read_agent_response(result)
