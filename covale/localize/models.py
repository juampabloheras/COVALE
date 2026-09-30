from pydantic import Field

from covale.utils.models import StrictModel


class SimilarityResponse(StrictModel):
    score: float = Field(ge=0, le=1)


class OverlapPrecheckResponse(StrictModel):
    confidently_disjoint: bool
    confidence: float = Field(ge=0, le=1)
    research_query: str | None = Field(default=None, min_length=1)
