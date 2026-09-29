from typing import Literal

from pydantic import Field, field_validator, model_validator

from covale.utils.models import StrictModel


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
