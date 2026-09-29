from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtlasRegion(StrictModel):
    mask: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class AtlasRegistry(StrictModel):
    atlas: str = Field(min_length=1)
    space: str = Field(min_length=1)
    regions: dict[str, AtlasRegion]


class RegionExpression(StrictModel):
    op: Literal["region"]
    id: str | None = Field(default=None, min_length=1)
    name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_one_reference(self) -> "RegionExpression":
        if (self.id is None) == (self.name is None):
            raise ValueError("Region expression requires exactly one of id or name.")
        return self


class UnionExpression(StrictModel):
    op: Literal["union"]
    args: list["Expression"] = Field(min_length=2)


class IntersectionExpression(StrictModel):
    op: Literal["intersection"]
    args: list["Expression"] = Field(min_length=2)


class DifferenceExpression(StrictModel):
    op: Literal["difference"]
    args: list["Expression"] = Field(min_length=2)


class UnresolvedExpression(StrictModel):
    op: Literal["unresolved"]


Expression = Annotated[
    RegionExpression | UnionExpression | IntersectionExpression | DifferenceExpression | UnresolvedExpression,
    Field(discriminator="op"),
]
UnionExpression.model_rebuild()
IntersectionExpression.model_rebuild()
DifferenceExpression.model_rebuild()
EXPRESSION_ADAPTER = TypeAdapter(Expression)


class Pair(BaseModel):
    model_config = ConfigDict(extra="allow")
    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)


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


class SimilarityResponse(StrictModel):
    score: float = Field(ge=0, le=1)


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
            raise ValueError("Successful benchmark record requires reference and candidate.")
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
    atlas_space: str | None
    registry: str
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


class DiceSettings(StrictModel):
    method: Literal["llm", "deep_agent", "similarity"] = "llm"
    provider: Literal["openai"] = "openai"
    model_name: str = "gpt-6-astra"
    registry_path: str | None = None
    concurrency: int | None = Field(default=None, ge=1)


class DiceMetric(StrictModel):
    dice: DiceSettings = Field(default_factory=DiceSettings)


MetricEntry = Literal["dice"] | DiceMetric


class OutputSettings(StrictModel):
    mode: Literal["default", "per_sample", "detailed"] = "default"
    errors: Literal["raise", "record"] = "raise"

    @model_validator(mode="after")
    def check_error_mode(self) -> "OutputSettings":
        if self.errors == "record" and self.mode != "detailed":
            raise ValueError("errors='record' requires mode='detailed'.")
        return self


class EvaluatorConfig(StrictModel):
    metrics: list[MetricEntry] = Field(min_length=1, max_length=1)
    extract_findings: bool = False
    output: OutputSettings = Field(default_factory=OutputSettings)
    cache: bool = True
    concurrency: int = Field(default=1, ge=1)

    @field_validator("metrics")
    @classmethod
    def require_dice(cls, metrics: list[MetricEntry]) -> list[MetricEntry]:
        if len(metrics) != 1:
            raise ValueError("Exactly one dice metric is required.")
        return metrics

    def dice_settings(self) -> DiceSettings:
        metric = self.metrics[0]
        return DiceSettings() if metric == "dice" else metric.dice


def validate_expression(value: object) -> Expression:
    return EXPRESSION_ADAPTER.validate_python(value)
