from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)
from covale.localize.models import SimilarityResponse
from covale.prompts import SIMILARITY_PROMPT


def compare(
    reference: str,
    candidate: str,
    *,
    client: OpenAIClient | None = None,
    model: str = "gpt-6-astra",
) -> float:
    client = client or create_client()
    result = request_json(
        client,
        model=model,
        instructions=SIMILARITY_PROMPT,
        payload={
            "reference": reference,
            "candidate": candidate,
        },
    )
    return SimilarityResponse.model_validate(result).score
