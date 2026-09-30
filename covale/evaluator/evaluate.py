import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import nibabel as nib
import numpy as np

from covale.evaluator.report_processing import (
    AnatomicalUnit,
    compatible_unit_pairs,
    extract_anatomical_units,
)
from covale.localize import (
    DEFAULT_REGISTRY,
    Expression,
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
from covale.localize.overlap import precheck_overlap
from covale.localize.similarity import compare
from covale.registry import AtlasMask, Registry
from covale.evaluator.metrics import dice
from covale.evaluator.report_processing.models import Alignment, UnitMatch

CovaleMethod = Literal["llm", "deep_agent", "similarity"]
DEFAULT_MODEL = "gpt-6-astra"
LocalizationCacheKey = tuple[str, str, str, str]
LocalizationCache = dict[
    LocalizationCacheKey,
    tuple[Expression, AtlasMask],
]


def _volume_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return name or "region"


def _volume_paths(
    units: list[AnatomicalUnit],
    output_dir: Path,
    prefix: str,
) -> dict[int, Path]:
    counts: dict[str, int] = {}
    paths = {}
    for unit in units:
        name = _volume_name(unit.anatomy)
        counts[name] = counts.get(name, 0) + 1
        suffix = f"_{counts[name]}" if counts[name] > 1 else ""
        paths[id(unit)] = output_dir / f"{prefix}_{name}{suffix}.nii.gz"
    return paths


def _save_volume(mask: AtlasMask, path: Path, value: int) -> None:
    data = np.where(mask.data, value, 0).astype(np.uint16)
    nib.save(nib.Nifti1Image(data, mask.affine), str(path))


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
    localization_cache: LocalizationCache,
    output_paths: tuple[Path, Path] | None,
) -> dict[str, Any]:
    started = perf_counter()
    if expression_builder is not None and method is not None:
        raise ValueError("Pass either expression_builder or method, not both.")

    builder = expression_builder
    overlap_precheck = None
    precheck_seconds = None
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

    if method == "deep_agent":
        precheck_started = perf_counter()
        precheck = precheck_overlap(
            reference,
            candidate,
            client=client,
            model=model,
        )
        precheck_seconds = perf_counter() - precheck_started
        overlap_precheck = precheck.model_dump(exclude_none=True)
        if precheck.confidently_disjoint and precheck.confidence >= 0.9:
            if output_paths is not None:
                warnings.warn(
                    "save_volumes was requested, but Deep Agent grounding "
                    "was skipped because the regions were confidently "
                    "disjoint; no volumes were written.",
                    UserWarning,
                    stacklevel=3,
                )
            return {
                "reference": reference,
                "candidate": candidate,
                "method": method,
                "score": 0.0,
                "diagnostics": {
                    "resolved": True,
                    "grounding_skipped": True,
                    "overlap_precheck": overlap_precheck,
                    "language_seconds": precheck_seconds,
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
        deep_agent = agent or create_agent(
            model,
            Registry.load(registry_path),
        )

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

    registry_key = str(Path(registry_path).resolve())
    selected_method = method or "custom"

    def localize_cached(text: str) -> tuple[Expression, AtlasMask]:
        key = (registry_key, selected_method, model, text)
        if key not in localization_cache:
            localization_cache[key] = localize_with_expression(
                text,
                registry_path,
                builder,
            )
        return localization_cache[key]

    reference_started = perf_counter()
    reference_expression, reference_mask = localize_cached(reference)
    reference_seconds = perf_counter() - reference_started
    candidate_started = perf_counter()
    candidate_expression, candidate_mask = localize_cached(candidate)
    candidate_seconds = perf_counter() - candidate_started
    saved_volumes = None
    if output_paths is not None:
        query_path, target_path = output_paths
        _save_volume(reference_mask, query_path, 100)
        _save_volume(candidate_mask, target_path, 200)
        saved_volumes = {
            "query": str(query_path.resolve()),
            "target": str(target_path.resolve()),
        }
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
            **(
                {
                    "grounding_skipped": False,
                    "overlap_precheck": overlap_precheck,
                    "precheck_seconds": precheck_seconds,
                }
                if overlap_precheck
                else {}
            ),
            **({"saved_volumes": saved_volumes} if saved_volumes else {}),
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
    save_volumes: str | Path | None = None,
    _localization_cache: LocalizationCache | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    if save_volumes is not None and method == "similarity":
        warnings.warn(
            "save_volumes is ignored for method='similarity' because "
            "language-only similarity does not produce atlas masks.",
            UserWarning,
            stacklevel=2,
        )
        save_volumes = None
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

    output_dir = Path(save_volumes) if save_volumes is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    query_paths = (
        _volume_paths(references, output_dir, "query") if output_dir else {}
    )
    target_paths = (
        _volume_paths(candidates, output_dir, "target") if output_dir else {}
    )
    localization_cache = (
        _localization_cache
        if _localization_cache is not None
        else {}
    )
    scored_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def score_pair(
        reference_unit: AnatomicalUnit,
        candidate_unit: AnatomicalUnit,
    ) -> dict[str, Any]:
        key = (reference_unit.anatomy, candidate_unit.anatomy)
        if key in scored_pairs and output_dir is None:
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
            localization_cache=localization_cache,
            output_paths=(
                query_paths[id(reference_unit)],
                target_paths[id(candidate_unit)],
            )
            if output_dir
            else None,
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
    save_volumes: str | Path | None = None,
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
        save_volumes=save_volumes,
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
