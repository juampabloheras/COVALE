from pathlib import Path
from pydantic import Field, ValidationError

from covale.localize.openai_client import (
    ModelResponseError,
    OpenAIClient,
    request_json,
)
from covale.models import AnatomicalUnit, ReportFinding, StrictModel

PROMPTS = Path(__file__).parent / "prompts"
EXTRACTION_PROMPT = PROMPTS / "extract.txt"
COMPATIBILITY_PROMPT = PROMPTS / "compatibility.txt"


class FindingExtraction(StrictModel):
    findings: list[ReportFinding]


class CompatiblePair(StrictModel):
    reference_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)


class CompatibilityResult(StrictModel):
    compatible_pairs: list[CompatiblePair]


def _prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Could not read prompt: {path}") from error


def extract_anatomical_units(
    report: str,
    *,
    client: OpenAIClient,
    model: str,
) -> list[AnatomicalUnit]:
    if not report.strip():
        raise ValueError("Report text must not be empty.")
    response = request_json(
        client,
        model=model,
        instructions=_prompt(EXTRACTION_PROMPT),
        payload={"report": report},
    )
    try:
        result = FindingExtraction.model_validate(response)
    except ValidationError as error:
        raise ModelResponseError(
            f"Invalid finding extraction response: {error}"
        ) from error
    return list(result.findings)


def compatible_unit_pairs(
    references: list[AnatomicalUnit],
    candidates: list[AnatomicalUnit],
    *,
    client: OpenAIClient,
    model: str,
) -> set[tuple[int, int]]:
    if not references or not candidates:
        return set()

    response = request_json(
        client,
        model=model,
        instructions=_prompt(COMPATIBILITY_PROMPT),
        payload={
            "reference_findings": [
                unit.model_dump() for unit in references
            ],
            "candidate_findings": [
                unit.model_dump() for unit in candidates
            ],
        },
    )
    try:
        result = CompatibilityResult.model_validate(response)
    except ValidationError as error:
        raise ModelResponseError(
            f"Invalid finding compatibility response: {error}"
        ) from error

    compatible: set[tuple[int, int]] = set()
    for pair in result.compatible_pairs:
        if pair.reference_index >= len(references):
            raise ModelResponseError(
                "Finding compatibility response contains an invalid "
                "reference index."
            )
        if pair.candidate_index >= len(candidates):
            raise ModelResponseError(
                "Finding compatibility response contains an invalid "
                "candidate index."
            )
        reference = references[pair.reference_index]
        candidate = candidates[pair.candidate_index]
        if reference.assertion != candidate.assertion:
            continue
        compatible.add((pair.reference_index, pair.candidate_index))
    return compatible
