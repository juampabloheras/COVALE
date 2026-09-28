import json
from collections.abc import Mapping
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


class ResponsesAPI(Protocol):
    def create(self, **request: object) -> object: ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class ModelResponseError(ValueError):
    pass


def create_client() -> OpenAI:
    load_dotenv()
    return OpenAI()


def request_json(
    client: OpenAIClient,
    *,
    model: str,
    instructions: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload),
    )
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str):
        raise ModelResponseError("OpenAI response did not contain text output.")

    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise ModelResponseError("OpenAI response was not valid JSON.") from error
    if not isinstance(result, dict):
        raise ModelResponseError("OpenAI response must be a JSON object.")
    return result
