from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from covale.evaluator.report_processing import (
    AnatomicalUnit,
    compatible_unit_pairs,
    extract_anatomical_units,
)
from covale.localize import (
    DEFAULT_REGISTRY,
    ExpressionBuilder,
    expression_stats,
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
from covale.evaluator.metrics import dice
from covale.evaluator.report_processing.models import Alignment, UnitMatch

CovaleMethod = Literal["llm", "deep_agent", "similarity"]
DEFAULT_MODEL = "gpt-6-astra"


def _evaluate_anatomy(
    reference: str,
    candidate: str,
    registry_path: str | Path,
    expression_builder: ExpressionBuilder | None,
    *,
    method: CovaleMethod | None,
    model: str,
    client: OpenAIClient | None,
    agent: DeepAgent | None,
) -> dict[str, Any]:
    started = perf_counter()
    if expression_builder is not None and method is not None:
        raise ValueError("Pass either expression_builder or method, not both.")

    builder = expression_builder
    if method == "similarity":
        score_started = perf_counter()
        score = compare(
            reference,
            candidate,
            client=client,
            model=model,
        )
        elapsed = perf_counter() - score_started
        return {
            "reference": reference,
            "candidate": candidate,
            "method": method,
            "score": score,
            "diagnostics": {
                "resolved": True,
                "language_seconds": elapsed,
                "total_seconds": perf_counter() - started,
            },
        }

    if method == "llm":
        openai_client = client or create_client()

        def build_with_llm(text: str, registry: Mapping[str, Any]) -> Mapping[str, Any]:
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

    reference_started = perf_counter()
    reference_expression, reference_mask = localize_with_expression(
        reference, registry_path, builder
    )
    reference_seconds = perf_counter() - reference_started
    candidate_started = perf_counter()
    candidate_expression, candidate_mask = localize_with_expression(
        candidate, registry_path, builder
    )
    candidate_seconds = perf_counter() - candidate_started
    dice_started = perf_counter()
    score = dice(reference_mask.data, candidate_mask.data)
    dice_seconds = perf_counter() - dice_started
    return {
        "reference": reference,
        "candidate": candidate,
        "method": method or "custom",
        "reference_expression": reference_expression,
        "candidate_expression": candidate_expression,
        "score": score,
        "diagnostics": {
            "resolved": True,
            "reference_localization_seconds": reference_seconds,
            "candidate_localization_seconds": candidate_seconds,
            "dice_seconds": dice_seconds,
            "total_seconds": perf_counter() - started,
            "reference_expression": expression_stats(reference_expression),
            "candidate_expression": expression_stats(candidate_expression),
        },
    }


def _minimum_cost_assignment(costs: list[list[float]]) -> list[int]:
    size = len(costs)
    if size == 0:
        return []
    potentials_left = [0.0] * (size + 1)
    potentials_right = [0.0] * (size + 1)
    matched_left = [0] * (size + 1)
    path = [0] * (size + 1)

    for left in range(1, size + 1):
        matched_left[0] = left
        right = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[right] = True
            current_left = matched_left[right]
            delta = float("inf")
            next_right = 0
            for candidate_right in range(1, size + 1):
                if used[candidate_right]:
                    continue
                reduced = (
                    costs[current_left - 1][candidate_right - 1]
                    - potentials_left[current_left]
                    - potentials_right[candidate_right]
                )
                if reduced < minimum[candidate_right]:
                    minimum[candidate_right] = reduced
                    path[candidate_right] = right
                if minimum[candidate_right] < delta:
                    delta = minimum[candidate_right]
                    next_right = candidate_right
            for candidate_right in range(size + 1):
                if used[candidate_right]:
                    potentials_left[matched_left[candidate_right]] += delta
                    potentials_right[candidate_right] -= delta
                else:
                    minimum[candidate_right] -= delta
            right = next_right
            if matched_left[right] == 0:
                break
        while True:
            previous = path[right]
            matched_left[right] = matched_left[previous]
            right = previous
            if right == 0:
                break

    assignment = [-1] * size
    for right in range(1, size + 1):
        if matched_left[right]:
            assignment[matched_left[right] - 1] = right - 1
    return assignment


def align_units(
    references: list[AnatomicalUnit],
    candidates: list[AnatomicalUnit],
    compatible: set[tuple[int, int]],
    score_pair: Callable[[AnatomicalUnit, AnatomicalUnit], dict[str, Any]],
) -> Alignment:
    reference_count = len(references)
    candidate_count = len(candidates)
    if not references or not candidates or not compatible:
        return Alignment(
            matches=[],
            missing=references,
            spurious=candidates,
        )

    pair_results: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in compatible:
        reference_index, candidate_index = pair
        pair_results[pair] = score_pair(
            references[reference_index],
            candidates[candidate_index],
        )

    size = reference_count + candidate_count
    weights = [[0.0] * size for _ in range(size)]
    for reference_index in range(reference_count):
        for candidate_index in range(candidate_count):
            pair = (reference_index, candidate_index)
            weights[reference_index][candidate_index] = (
                float(pair_results[pair]["score"]) + 1e-9
                if pair in pair_results
                else -2.0
            )
    assignment = _minimum_cost_assignment(
        [[-weight for weight in row] for row in weights]
    )

    matches: list[UnitMatch] = []
    matched_references: set[int] = set()
    matched_candidates: set[int] = set()
    for reference_index in range(reference_count):
        candidate_index = assignment[reference_index]
        pair = (reference_index, candidate_index)
        if candidate_index >= candidate_count or pair not in pair_results:
            continue
        result = pair_results[pair]
        matches.append(
            UnitMatch(
                reference_index=reference_index,
                candidate_index=candidate_index,
                reference=references[reference_index],
                candidate=candidates[candidate_index],
                score=result["score"],
                diagnostics=result["diagnostics"],
            )
        )
        matched_references.add(reference_index)
        matched_candidates.add(candidate_index)

    return Alignment(
        matches=matches,
        missing=[
            unit
            for index, unit in enumerate(references)
            if index not in matched_references
        ],
        spurious=[
            unit
            for index, unit in enumerate(candidates)
            if index not in matched_candidates
        ],
    )


def evaluate(
    reference: str,
    candidate: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    expression_builder: ExpressionBuilder | None = None,
    *,
    method: CovaleMethod | None = None,
    model: str = DEFAULT_MODEL,
    extract_findings: bool = False,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    if extract_findings:
        openai_client = client or create_client()
        client = openai_client
        references = extract_anatomical_units(
            reference,
            client=openai_client,
            model=model,
        )
        candidates = extract_anatomical_units(
            candidate,
            client=openai_client,
            model=model,
        )
        compatible = compatible_unit_pairs(
            references,
            candidates,
            client=openai_client,
            model=model,
        )
    else:
        references = [AnatomicalUnit(text=reference, anatomy=reference)]
        candidates = [AnatomicalUnit(text=candidate, anatomy=candidate)]
        compatible = {(0, 0)}

    scored_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def score_pair(
        reference_unit: AnatomicalUnit,
        candidate_unit: AnatomicalUnit,
    ) -> dict[str, Any]:
        key = (reference_unit.anatomy, candidate_unit.anatomy)
        if key in scored_pairs:
            return scored_pairs[key]
        result = _evaluate_anatomy(
            reference_unit.anatomy,
            candidate_unit.anatomy,
            registry_path,
            expression_builder,
            method=method,
            model=model,
            client=client,
            agent=agent,
        )
        scored_pairs[key] = result
        return result

    alignment = align_units(
        references,
        candidates,
        compatible,
        score_pair,
    )
    if not extract_findings:
        return score_pair(references[0], candidates[0])

    denominator = max(len(references), len(candidates))
    score = (
        sum(match.score for match in alignment.matches) / denominator
        if denominator
        else 1.0
    )
    return {
        "reference": reference,
        "candidate": candidate,
        "method": method or "custom",
        "score": score,
        "diagnostics": {
            "resolved": True,
            "extract_findings": True,
            "reference_findings": [unit.model_dump() for unit in references],
            "candidate_findings": [unit.model_dump() for unit in candidates],
            **alignment.model_dump(mode="json"),
            "total_seconds": perf_counter() - started,
        },
    }


def covale(
    reference: str,
    candidate: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    method: CovaleMethod = "llm",
    model: str = DEFAULT_MODEL,
    extract_findings: bool = False,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> float:
    result = evaluate(
        reference,
        candidate,
        registry_path,
        method=method,
        model=model,
        extract_findings=extract_findings,
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
    extract_findings: bool = False,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> list[dict[str, Any]]:
    if (extract_findings or method in {"llm", "similarity"}) and client is None:
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
                extract_findings=extract_findings,
                client=client,
                agent=agent,
            )
        )
    return results
