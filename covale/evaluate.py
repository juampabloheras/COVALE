from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from covale.localize import (
    DEFAULT_REGISTRY,
    ExpressionBuilder,
    localize_with_expression,
)
from covale.localize.deep_agent import (
    DeepAgent,
    build_expression as agent_expression,
    create_agent,
)
from covale.localize.llm import build_expression as llm_expression
from covale.localize.openai_client import OpenAIClient, create_client
from covale.localize.similarity import compare
from covale.metrics import dice

CovaleMethod = Literal["llm", "deep_agent", "similarity"]
DEFAULT_MODEL = "gpt-6-astra"


def evaluate(
    reference: str,
    candidate: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    expression_builder: ExpressionBuilder | None = None,
    *,
    method: CovaleMethod | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> dict[str, Any]:
    if expression_builder is not None and method is not None:
        raise ValueError("Pass either expression_builder or method, not both.")

    builder = expression_builder
    if method == "similarity":
        return {
            "reference": reference,
            "candidate": candidate,
            "method": method,
            "score": compare(
                reference,
                candidate,
                client=client,
                model=model,
            ),
        }

    if method == "llm":
        openai_client = client or create_client()

        def build_with_llm(
            text: str, registry: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return llm_expression(
                text,
                registry,
                client=openai_client,
                model=model,
            )

        builder = build_with_llm

    elif method == "deep_agent":
        deep_agent = agent or create_agent(model)

        def build_with_agent(
            text: str, registry: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            return agent_expression(
                text,
                registry,
                agent=deep_agent,
                model=model,
            )

        builder = build_with_agent

    elif method is not None:
        raise ValueError(f"Unknown COVALE method: {method}")

    reference_expression, reference_mask = localize_with_expression(
        reference, registry_path, builder
    )
    candidate_expression, candidate_mask = localize_with_expression(
        candidate, registry_path, builder
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "method": method or "custom",
        "reference_expression": reference_expression,
        "candidate_expression": candidate_expression,
        "score": dice(reference_mask.data, candidate_mask.data),
    }


def covale(
    reference: str,
    candidate: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    method: CovaleMethod = "llm",
    model: str = DEFAULT_MODEL,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> float:
    result = evaluate(
        reference,
        candidate,
        registry_path,
        method=method,
        model=model,
        client=client,
        agent=agent,
    )
    return float(result["score"])


def evaluate_batch(
    pairs: Iterable[Mapping[str, object]],
    registry_path: str | Path = DEFAULT_REGISTRY,
    expression_builder: ExpressionBuilder | None = None,
    *,
    method: CovaleMethod | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> list[dict[str, Any]]:
    if method in {"llm", "similarity"} and client is None:
        client = create_client()
    if method == "deep_agent" and agent is None:
        agent = create_agent(model)

    results = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"Pair {index} must be a JSON object.")
        reference = pair.get("reference")
        candidate = pair.get("candidate")
        if not isinstance(reference, str) or not isinstance(candidate, str):
            raise ValueError(
                f"Pair {index} must contain string reference and candidate fields."
            )
        results.append(
            evaluate(
                reference,
                candidate,
                registry_path,
                expression_builder,
                method=method,
                model=model,
                client=client,
                agent=agent,
            )
        )
    return results
