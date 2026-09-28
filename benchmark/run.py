import argparse
import hashlib
import json
import platform
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from covale import evaluate
from covale.evaluate import DEFAULT_MODEL, CovaleMethod
from covale.localize import LOCALIZATION_PROMPT
from covale.localize.deep_agent import DeepAgent, create_agent
from covale.localize.openai_client import OpenAIClient, create_client
from covale.models import (
    AtlasRegistry,
    BenchmarkRecord,
    BenchmarkSummary,
    ErrorAnnotation,
    Pair,
    SuccessfulAnnotation,
)

EXPECTED_ROW_ERRORS = (
    ValueError,
    RuntimeError,
    TimeoutError,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: str | Path) -> str | None:
    git_directory = Path(repository) / ".git"
    head = git_directory / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value
    reference = git_directory / value.removeprefix("ref: ")
    return reference.read_text(encoding="utf-8").strip()


def annotate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: CovaleMethod = "llm",
    model: str = DEFAULT_MODEL,
    registry_path: str | Path = "atlas_registry/registry.json",
    overwrite: bool = False,
    concurrency: int = 1,
    summary_path: str | Path | None = None,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    registry_path = Path(registry_path)
    summary_path = (
        Path(summary_path)
        if summary_path is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")
    for destination in (output_path, summary_path):
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. "
                "Pass --overwrite to replace it."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if method in {"llm", "similarity"} and client is None:
        client = create_client()
    if method == "deep_agent" and agent is None:
        agent = create_agent(model)
    setup_seconds = perf_counter() - started

    lines = [
        (line_number, line)
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.strip()
    ]

    def annotate_line(item: tuple[int, str]) -> tuple[dict[str, Any], float, bool]:
        line_number, line = item
        row_started = perf_counter()
        parsed: object = None
        try:
            pair = Pair.model_validate_json(line)
            parsed = pair.model_dump()

            result = evaluate(
                pair.reference,
                pair.candidate,
                registry_path,
                method=method,
                model=model,
                client=client,
                agent=agent,
            )
            elapsed = perf_counter() - row_started
            annotation = BenchmarkRecord.model_validate(
                {
                    **parsed,
                    "covale": SuccessfulAnnotation(
                    status="ok",
                    score=result["score"],
                    elapsed_seconds=elapsed,
                    method=method,
                    model=model,
                    line=line_number,
                    diagnostics=result["diagnostics"],
                ),
                }
            )
            return annotation.model_dump(mode="json"), elapsed, True
        except EXPECTED_ROW_ERRORS as error:
            elapsed = perf_counter() - row_started
            original = (
                parsed
                if isinstance(parsed, dict)
                else {"source": line}
            )
            annotation = BenchmarkRecord.model_validate(
                {
                    **original,
                    "covale": ErrorAnnotation(
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                    elapsed_seconds=elapsed,
                    method=method,
                    model=model,
                    line=line_number,
                ),
                }
            )
            return annotation.model_dump(mode="json"), elapsed, False

    if concurrency == 1:
        annotated_rows = [annotate_line(line) for line in lines]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            annotated_rows = list(executor.map(annotate_line, lines))

    with output_path.open("w", encoding="utf-8") as destination:
        for annotation, _, _ in annotated_rows:
            destination.write(json.dumps(annotation) + "\n")

    durations = [elapsed for _, elapsed, _ in annotated_rows]
    succeeded = sum(success for _, _, success in annotated_rows)
    failed = len(annotated_rows) - succeeded
    total_seconds = perf_counter() - started

    atlas: str | None = None
    space: str | None = None
    registry_hash: str | None = None
    if registry_path.is_file():
        registry = AtlasRegistry.model_validate_json(
            registry_path.read_text(encoding="utf-8")
        )
        atlas = registry.atlas
        space = registry.space
        registry_hash = file_sha256(registry_path)

    repository = Path(__file__).resolve().parent.parent
    summary = BenchmarkSummary.model_validate({
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "summary": str(summary_path),
        "method": method,
        "model": model,
        "concurrency": concurrency,
        "atlas": atlas,
        "atlas_space": space,
        "registry": str(registry_path),
        "registry_sha256": registry_hash,
        "localization_prompt_sha256": file_sha256(LOCALIZATION_PROMPT),
        "covale_version": version("covale"),
        "git_commit": git_commit(repository),
        "python_version": platform.python_version(),
        "rows": len(annotated_rows),
        "succeeded": succeeded,
        "failed": failed,
        "setup_seconds": setup_seconds,
        "total_seconds": total_seconds,
        "mean_row_seconds": (
            sum(durations) / len(durations) if durations else 0.0
        ),
    })
    summary_path.write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return summary.model_dump(mode="json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate JSONL phrase pairs with COVALE scores and timings."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument("output", type=Path, help="Annotated output JSONL file.")
    parser.add_argument(
        "--method",
        choices=("llm", "deep_agent", "similarity"),
        default="llm",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("atlas_registry/registry.json"),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = annotate_jsonl(
        args.input,
        args.output,
        method=args.method,
        model=args.model,
        registry_path=args.registry,
        concurrency=args.concurrency,
        summary_path=args.summary,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
