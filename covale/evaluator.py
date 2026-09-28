from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import yaml

from covale.evaluate import DEFAULT_MODEL, CovaleMethod, covale as score_covale
from covale.localize import DEFAULT_REGISTRY
from covale.localize.deep_agent import DeepAgent, create_agent
from covale.localize.openai_client import OpenAIClient, create_client

OutputMode = Literal["default", "per_sample", "detailed"]


class COVALE:
    def __init__(
        self,
        metrics: Sequence[str] = ("dice",),
        *,
        method: CovaleMethod = "llm",
        model: str = DEFAULT_MODEL,
        registry_path: str | Path = DEFAULT_REGISTRY,
        provider: Literal["openai"] = "openai",
        per_sample: bool = False,
        output_mode: OutputMode | None = None,
        cache: bool = True,
        client: OpenAIClient | None = None,
        agent: DeepAgent | None = None,
    ) -> None:
        if list(metrics) != ["dice"]:
            raise ValueError("COVALE supports only metrics=['dice'].")
        if method not in {"llm", "deep_agent", "similarity"}:
            raise ValueError(f"Unknown COVALE method: {method}")
        if provider != "openai":
            raise ValueError("COVALE currently supports only provider='openai'.")
        if output_mode not in {None, "default", "per_sample", "detailed"}:
            raise ValueError(f"Unknown output mode: {output_mode}")
        if per_sample and output_mode not in {None, "per_sample"}:
            raise ValueError("per_sample=True conflicts with output_mode.")

        self.method = method
        self.model = model
        self.registry_path = registry_path
        self.provider = provider
        self.output_mode = output_mode or (
            "per_sample" if per_sample else "default"
        )
        self.cache = cache
        self.client = client
        self.agent = agent
        self.scores: dict[tuple[str, str, str, str], float] = {}

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        client: OpenAIClient | None = None,
        agent: DeepAgent | None = None,
    ) -> "COVALE":
        config_path = Path(path)
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Could not read config: {config_path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML config: {config_path}") from error

        if not isinstance(config, Mapping):
            raise ValueError("Config must be a YAML mapping.")
        unknown_sections = set(config) - {"metrics", "output", "cache"}
        if unknown_sections:
            raise ValueError(
                f"Unknown config sections: {sorted(unknown_sections)}"
            )

        metrics = config.get("metrics")
        if not isinstance(metrics, list) or len(metrics) != 1:
            raise ValueError("Config metrics must contain exactly one dice entry.")

        metric_entry = metrics[0]
        if metric_entry == "dice":
            settings: Mapping[str, object] = {}
        elif (
            isinstance(metric_entry, Mapping)
            and set(metric_entry) == {"dice"}
            and isinstance(metric_entry["dice"], Mapping)
        ):
            settings = metric_entry["dice"]
        else:
            raise ValueError("Config supports only the dice metric.")

        allowed_settings = {
            "method",
            "provider",
            "model_name",
            "registry_path",
        }
        unknown_settings = set(settings) - allowed_settings
        if unknown_settings:
            raise ValueError(
                f"Unknown dice settings: {sorted(unknown_settings)}"
            )

        method = settings.get("method", "llm")
        provider = settings.get("provider", "openai")
        model = settings.get("model_name", DEFAULT_MODEL)
        registry = settings.get("registry_path", str(DEFAULT_REGISTRY))
        if not isinstance(method, str):
            raise ValueError("dice.method must be a string.")
        if not isinstance(provider, str):
            raise ValueError("dice.provider must be a string.")
        if not isinstance(model, str):
            raise ValueError("dice.model_name must be a string.")
        if not isinstance(registry, str):
            raise ValueError("dice.registry_path must be a string.")
        if method not in {"llm", "deep_agent", "similarity"}:
            raise ValueError(f"Unknown COVALE method: {method}")
        if provider != "openai":
            raise ValueError("COVALE currently supports only provider='openai'.")
        method = cast(CovaleMethod, method)
        provider = cast(Literal["openai"], provider)

        registry_path = Path(registry)
        if not registry_path.is_absolute():
            registry_path = (config_path.parent / registry_path).resolve()

        output = config.get("output", {})
        if not isinstance(output, Mapping):
            raise ValueError("output must be a mapping.")
        unknown_output = set(output) - {"mode"}
        if unknown_output:
            raise ValueError(f"Unknown output settings: {sorted(unknown_output)}")
        output_mode = output.get("mode", "default")
        if not isinstance(output_mode, str):
            raise ValueError("output.mode must be a string.")
        if output_mode not in {"default", "per_sample", "detailed"}:
            raise ValueError(f"Unknown output mode: {output_mode}")
        output_mode = cast(OutputMode, output_mode)

        cache = config.get("cache", True)
        if not isinstance(cache, bool):
            raise ValueError("cache must be true or false.")

        return cls(
            metrics=["dice"],
            method=method,
            model=model,
            registry_path=registry_path,
            provider=provider,
            output_mode=output_mode,
            cache=cache,
            client=client,
            agent=agent,
        )

    def score(self, reference: str, candidate: str) -> float:
        key = (self.method, self.model, reference, candidate)
        if self.cache and key in self.scores:
            return self.scores[key]

        if self.method in {"llm", "similarity"} and self.client is None:
            self.client = create_client()
        if self.method == "deep_agent" and self.agent is None:
            self.agent = create_agent(self.model)

        score = score_covale(
            reference,
            candidate,
            self.registry_path,
            method=self.method,
            model=self.model,
            client=self.client,
            agent=self.agent,
        )
        if self.cache:
            self.scores[key] = score
        return score

    def __call__(
        self,
        refs: Sequence[str],
        hyps: Sequence[str],
    ) -> dict[str, float | list[float]]:
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps must have the same length.")
        if not refs:
            raise ValueError("refs and hyps must not be empty.")

        scores = [
            self.score(reference, candidate)
            for reference, candidate in zip(refs, hyps, strict=True)
        ]
        if self.output_mode == "per_sample":
            return {"dice": scores}
        mean = float(np.mean(scores))
        if self.output_mode == "detailed":
            return {
                "dice": mean,
                "dice_std": float(np.std(scores)),
                "dice_per_sample": scores,
            }
        return {"dice": mean}
