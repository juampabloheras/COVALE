from pydantic import Field

from covale.utils.models import StrictModel


class SimilarityResponse(StrictModel):
    score: float = Field(ge=0, le=1)
