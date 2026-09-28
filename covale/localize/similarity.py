from pathlib import Path

from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)
from covale.models import SimilarityResponse

SIMILARITY_PROMPT = Path(__file__).parent / "prompts" / "similarity.txt"


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
        instructions=SIMILARITY_PROMPT.read_text(encoding="utf-8"),
        payload={
            "reference": reference,
            "candidate": candidate,
        },
    )
    return SimilarityResponse.model_validate(result).score
