from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import ValidationError

from covale.evaluator.evaluate import (
    DEFAULT_MODEL,
    CovaleMethod,
    evaluate as evaluate_pair,
)
from covale.localize import DEFAULT_REGISTRY
from covale.localize.deep_agent import DeepAgent, create_agent
from covale.localize.openai_client import OpenAIClient, create_client
from covale.evaluator.models import EvaluatorConfig

OutputMode = Literal["default", "per_sample", "detailed"]
ErrorMode = Literal["raise", "record"]


class COVALE:
    def __init__(
        self,
        metrics: Sequence[str] = ("dice",),
        *,
        method: CovaleMethod = "llm",
        model: str = DEFAULT_MODEL,
        registry_path: str | Path = DEFAULT_REGISTRY,
        provider: Literal["openai"] = "openai",
        extract_findings: bool = False,
        per_sample: bool = False,
        output_mode: OutputMode | None = None,
        cache: bool = True,
        concurrency: int = 1,
        errors: ErrorMode = "raise",
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
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1.")
        if errors not in {"raise", "record"}:
            raise ValueError(f"Unknown error mode: {errors}")
        selected_output = output_mode or ("per_sample" if per_sample else "default")
        if errors == "record" and selected_output != "detailed":
            raise ValueError("errors='record' requires output_mode='detailed'.")

        self.method = method
        self.model = model
        self.registry_path = registry_path
        self.provider = provider
        self.extract_findings = extract_findings
        self.output_mode = selected_output
        self.cache = cache
        self.concurrency = concurrency
        self.errors = errors
        self.client = client
        self.agent = agent
        self.results: dict[tuple[str, str, bool, str, str], dict[str, Any]] = {}

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
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config = EvaluatorConfig.model_validate(config_data)
        except OSError as error:
            raise ValueError(f"Could not read config: {config_path}") from error
        except (yaml.YAMLError, ValidationError) as error:
            raise ValueError(f"Invalid YAML config {config_path}: {error}") from error

        settings = config.dice_settings()
        registry_path = Path(settings.registry_path or str(DEFAULT_REGISTRY))
        if not registry_path.is_absolute():
            registry_path = (config_path.parent / registry_path).resolve()

        return cls(
            metrics=["dice"],
            method=settings.method,
            model=settings.model_name,
            registry_path=registry_path,
            provider=settings.provider,
            extract_findings=config.extract_findings,
            output_mode=config.output.mode,
            cache=config.cache,
            concurrency=settings.concurrency or config.concurrency,
            errors=config.output.errors,
            client=client,
            agent=agent,
        )

    def prepare_provider(self) -> None:
        if (
            self.extract_findings or self.method in {"llm", "similarity"}
        ) and self.client is None:
            self.client = create_client()
        if self.method == "deep_agent" and self.agent is None:
            self.agent = create_agent(self.model)

    def evaluate_pair(
        self,
        reference: str,
        candidate: str,
    ) -> dict[str, Any]:
        key = (
            self.method,
            self.model,
            self.extract_findings,
            reference,
            candidate,
        )
        if self.cache and key in self.results:
            return self.results[key]

        self.prepare_provider()
        try:
            result = evaluate_pair(
                reference,
                candidate,
                self.registry_path,
                method=self.method,
                model=self.model,
                extract_findings=self.extract_findings,
                client=self.client,
                agent=self.agent,
            )
        except (ValueError, RuntimeError, TimeoutError) as error:
            if self.errors == "raise":
                raise
            result = {
                "reference": reference,
                "candidate": candidate,
                "method": self.method,
                "score": None,
                "diagnostics": {
                    "resolved": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            }
        if self.cache:
            self.results[key] = result
        return result

    def score(self, reference: str, candidate: str) -> float:
        result = self.evaluate_pair(reference, candidate)
        score = result.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("COVALE pair did not produce a Dice score.")
        return float(score)

    def __call__(
        self,
        refs: Sequence[str],
        hyps: Sequence[str],
    ) -> dict[str, Any]:
        if len(refs) != len(hyps):
            raise ValueError("refs and hyps must have the same length.")
        if not refs:
            raise ValueError("refs and hyps must not be empty.")

        pairs = list(zip(refs, hyps, strict=True))
        self.prepare_provider()
        work = list(dict.fromkeys(pairs)) if self.cache else pairs

        if self.concurrency == 1:
            work_results = [
                self.evaluate_pair(reference, candidate)
                for reference, candidate in work
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                work_results = list(
                    executor.map(
                        lambda pair: self.evaluate_pair(*pair),
                        work,
                    )
                )

        if self.cache:
            by_pair = dict(zip(work, work_results, strict=True))
            results = [by_pair[pair] for pair in pairs]
        else:
            results = work_results

        scores = [result.get("score") for result in results]
        if self.output_mode == "per_sample":
            return {"dice": scores}
        resolved_scores = [
            float(score)
            for score in scores
            if isinstance(score, int | float) and not isinstance(score, bool)
        ]
        if not resolved_scores:
            raise ValueError("No COVALE pairs were resolved.")
        mean = float(np.mean(resolved_scores))
        if self.output_mode == "detailed":
            failures = [
                result["diagnostics"]
                for result in results
                if not result["diagnostics"]["resolved"]
            ]
            failure_counts = Counter(failure["error_type"] for failure in failures)
            return {
                "dice": mean,
                "dice_std": float(np.std(resolved_scores)),
                "dice_per_sample": scores,
                "resolved_fraction": len(resolved_scores) / len(results),
                "resolved_count": len(resolved_scores),
                "unresolved_count": len(failures),
                "failure_counts": dict(failure_counts),
                "pairs": results,
            }
        return {"dice": mean}
