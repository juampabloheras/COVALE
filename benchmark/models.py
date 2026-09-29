from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from covale.utils.models import StrictModel


class Pair(BaseModel):
    model_config = ConfigDict(extra="allow")
    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)


class SuccessfulAnnotation(StrictModel):
    status: Literal["ok"]
    score: float = Field(ge=0, le=1)
    elapsed_seconds: float = Field(ge=0)
    method: Literal["llm", "deep_agent", "similarity"]
    model: str
    line: int = Field(ge=1)
    diagnostics: dict[str, Any]


class ErrorAnnotation(StrictModel):
    status: Literal["error"]
    error_type: str
    error: str
    elapsed_seconds: float = Field(ge=0)
    method: Literal["llm", "deep_agent", "similarity"]
    model: str
    line: int = Field(ge=1)


Annotation = Annotated[
    SuccessfulAnnotation | ErrorAnnotation,
    Field(discriminator="status"),
]


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    reference: str | None = None
    candidate: str | None = None
    source: str | None = None
    covale: Annotation

    @model_validator(mode="after")
    def check_source(self) -> "BenchmarkRecord":
        has_pair = self.reference is not None and self.candidate is not None
        if not has_pair and self.source is None:
            raise ValueError("Benchmark record requires a phrase pair or source text.")
        if self.covale.status == "ok" and not has_pair:
            raise ValueError(
                "Successful benchmark record requires reference and candidate."
            )
        return self


class BenchmarkSummary(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    created_at: str
    input: str
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    output: str
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    summary: str
    method: Literal["llm", "deep_agent", "similarity"]
    model: str
    concurrency: int = Field(ge=1)
    atlas: str | None
    atlas_ids: list[str] = Field(default_factory=list)
    atlas_space: str | None
    registry: str
    registry_schema_version: Literal[1, 2] | None = None
    registry_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    localization_prompt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extraction_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    compatibility_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    extract_findings: bool
    covale_version: str
    git_commit: str | None
    python_version: str
    rows: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    setup_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)
    mean_row_seconds: float = Field(ge=0)
