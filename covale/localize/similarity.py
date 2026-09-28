from pathlib import Path

from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)

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
    score = result.get("score")
    if not isinstance(score, int | float) or isinstance(score, bool):
        raise ValueError("Similarity response must contain a numeric score.")
    if not 0 <= score <= 1:
        raise ValueError("Similarity score must be between 0 and 1.")
    return float(score)
