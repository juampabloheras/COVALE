from typing import Any, Literal

from pydantic import Field

from covale.utils.models import StrictModel


class AnatomicalUnit(StrictModel):
    text: str = Field(min_length=1)
    anatomy: str = Field(min_length=1)
    concept: str | None = None
    assertion: Literal["present", "absent", "uncertain"] | None = None


class ReportFinding(AnatomicalUnit):
    concept: str = Field(min_length=1)
    assertion: Literal["present", "absent", "uncertain"]


class UnitMatch(StrictModel):
    reference_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    reference: AnatomicalUnit
    candidate: AnatomicalUnit
    score: float = Field(ge=0, le=1)
    diagnostics: dict[str, Any]


class Alignment(StrictModel):
    matches: list[UnitMatch]
    missing: list[AnatomicalUnit]
    spurious: list[AnatomicalUnit]
