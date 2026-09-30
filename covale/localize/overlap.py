import os

from covale.localize.models import OverlapPrecheckResponse
from covale.localize.openai_client import (
    OpenAIClient,
    create_client,
    request_json,
)
from covale.localize.registry_tools import tavily_search
from covale.prompts import OVERLAP_PRECHECK_PROMPT


def precheck_overlap(
    reference: str,
    candidate: str,
    *,
    client: OpenAIClient | None = None,
    model: str = "gpt-6-astra",
) -> OverlapPrecheckResponse:
    """Conservatively identify clearly spatially disjoint anatomy."""
    client = client or create_client()
    payload: dict[str, object] = {
        "reference": reference,
        "candidate": candidate,
    }
    result = request_json(
        client,
        model=model,
        instructions=OVERLAP_PRECHECK_PROMPT,
        payload=payload,
    )
    precheck = OverlapPrecheckResponse.model_validate(result)
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if precheck.research_query and api_key:
        payload["web_research"] = tavily_search(
            precheck.research_query,
            api_key=api_key,
        )
        result = request_json(
            client,
            model=model,
            instructions=OVERLAP_PRECHECK_PROMPT,
            payload=payload,
        )
        precheck = OverlapPrecheckResponse.model_validate(result)
    return precheck
