import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from covale.localize.atlas_expression import (
    Expression,
    LocalizationError,
)
from covale.localize.openai_client import ProviderError
from covale.localize.registry_tools import (
    RegistryLocalizationTools,
    tavily_search,
)
from covale.prompts import LOCALIZATION_PROMPT
from covale.registry import Registry
from covale.registry.models import RegistryV2

DEEP_AGENT_PROMPT = f"""
{LOCALIZATION_PROMPT}

You can iteratively search and browse the complete atlas registry. Decompose
broad or absent anatomy into spatially appropriate components. Search each
component separately, inspect uncertain results, and use anatomy_web_search
only when atlas metadata is insufficient to determine the decomposition.
External sources may inform anatomy but never supply region IDs.
Stable IDs returned by registry tools are supplied atlas IDs and may be used
even when they were absent from the initial candidate catalog.

When the request names a subpart that has no atlas label, research its
anatomical boundaries and construct it with the documented geometric
primitives. Use directional_part for relative directional portions,
clip_plane for literature-supported RAS boundaries, and dilate or erode only
when a morphological approximation is justified. Call
inspect_expression_geometry before choosing an explicit plane offset. Preserve
the parent region as the primitive's input so the result cannot extend outside
that anatomy unless deliberate dilation is required. Treat the result as an
approximation and choose reproducible numeric parameters.

Before returning, call validate_atlas_expression with the exact proposed
expression. Return only that validated expression as JSON. If no valid
representation is possible, return {{"op": "unresolved"}}.
""".strip()


class DeepAgent(Protocol):
    def invoke(self, request: dict[str, object]) -> Mapping[str, Any]: ...


def create_agent(
    model: str = "gpt-6-astra",
    registry: Registry | None = None,
) -> DeepAgent:
    try:
        from deepagents import create_deep_agent
        from dotenv import load_dotenv
        from langchain.agents.middleware import (
            ModelCallLimitMiddleware,
            ToolCallLimitMiddleware,
        )
        from openai import OpenAIError
    except ImportError as error:
        raise ProviderError(
            "Deep Agents is a core COVALE dependency. Reinstall COVALE to "
            "restore the provider."
        ) from error

    load_dotenv()
    provider_model = model if ":" in model else f"openai:{model}"
    tools = RegistryLocalizationTools(registry).tool_functions() if registry else []
    middleware = [
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=12, exit_behavior="error"),
        ToolCallLimitMiddleware(
            tool_name="validate_atlas_expression",
            run_limit=3,
            exit_behavior="error",
        ),
    ]
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and registry is not None:

        def anatomy_web_search(
            query: str,
            max_results: int = 5,
        ) -> dict[str, object]:
            """Search anatomy references when registry metadata is insufficient."""
            return tavily_search(
                query,
                api_key=api_key,
                max_results=max_results,
            )

        tools.append(anatomy_web_search)
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="anatomy_web_search",
                run_limit=3,
                exit_behavior="error",
            )
        )
    try:
        return create_deep_agent(
            model=provider_model,
            tools=tools,
            system_prompt=DEEP_AGENT_PROMPT,
            middleware=middleware,
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
        raise LocalizationError("Deep agent response was not valid JSON.") from error
    if not isinstance(expression, dict) or not isinstance(expression.get("op"), str):
        raise LocalizationError("Deep agent response must contain a JSON expression with an op.")
    return expression


def build_expression(
    text: str,
    registry: Mapping[str, Any],
    *,
    agent: DeepAgent | None = None,
    model: str = "gpt-6-astra",
) -> Expression:
    try:
        from openai import OpenAIError
    except ImportError as error:
        raise ProviderError(
            "Deep Agents is a core COVALE dependency. Reinstall COVALE to "
            "restore the provider."
        ) from error

    try:
        registry_model = RegistryV2.model_validate(registry)
        loaded_registry = Registry(
            registry_model,
            Path.cwd(),
        )
        active_agent = agent or create_agent(model, loaded_registry)
        candidates = loaded_registry.composition_catalog(text)
        result = active_agent.invoke(
            {
                "messages": json.dumps(
                    {
                        "anatomical_description": text,
                        "candidate_regions": candidates,
                    }
                )
            }
        )
    except (OpenAIError, RuntimeError) as error:
        raise ProviderError(f"Deep Agent request failed: {error}") from error
    return read_agent_response(result)
