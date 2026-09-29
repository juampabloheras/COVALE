import json
from collections.abc import Mapping
from typing import Protocol


class ResponsesAPI(Protocol):
    def create(self, **request: object) -> object: ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class ModelResponseError(ValueError):
    pass


class ProviderError(RuntimeError):
    pass


def create_client() -> OpenAIClient:
    try:
        from dotenv import load_dotenv
        from openai import OpenAI, OpenAIError
    except ImportError as error:
        raise ProviderError("OpenAI is a core COVALE dependency. Reinstall COVALE to " "restore the provider.") from error

    load_dotenv()
    try:
        return OpenAI()
    except OpenAIError as error:
        raise ProviderError(f"Could not initialize OpenAI: {error}") from error


def request_json(
    client: OpenAIClient,
    *,
    model: str,
    instructions: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    try:
        from openai import OpenAIError
    except ImportError as error:
        raise ProviderError("OpenAI is a core COVALE dependency. Reinstall COVALE to " "restore the provider.") from error

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(payload),
        )
    except OpenAIError as error:
        raise ProviderError(f"OpenAI request failed: {error}") from error
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
