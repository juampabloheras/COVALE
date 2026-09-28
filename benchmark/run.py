import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAIError

from covale import covale
from covale.evaluate import DEFAULT_MODEL, CovaleMethod
from covale.localize.deep_agent import DeepAgent, create_agent
from covale.localize.openai_client import OpenAIClient, create_client

EXPECTED_ROW_ERRORS = (
    ValueError,
    OpenAIError,
    RuntimeError,
    TimeoutError,
)


def annotate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: CovaleMethod = "llm",
    model: str = DEFAULT_MODEL,
    registry_path: str | Path = "atlas_registry/registry.json",
    overwrite: bool = False,
    client: OpenAIClient | None = None,
    agent: DeepAgent | None = None,
) -> dict[str, int | float | str]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if method in {"llm", "similarity"} and client is None:
        client = create_client()
    if method == "deep_agent" and agent is None:
        agent = create_agent(model)
    setup_seconds = perf_counter() - started

    rows = 0
    succeeded = 0
    failed = 0
    measured_seconds = 0.0

    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            rows += 1
            row_started = perf_counter()
            parsed: object = None
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("JSONL row must be an object.")

                reference = parsed.get("reference")
                candidate = parsed.get("candidate")
                if not isinstance(reference, str) or not isinstance(candidate, str):
                    raise ValueError(
                        "JSONL row must contain string reference and candidate fields."
                    )

                score = covale(
                    reference,
                    candidate,
                    registry_path,
                    method=method,
                    model=model,
                    client=client,
                    agent=agent,
                )
                elapsed = perf_counter() - row_started
                measured_seconds += elapsed
                succeeded += 1
                parsed["covale"] = {
                    "status": "ok",
                    "score": score,
                    "elapsed_seconds": elapsed,
                    "method": method,
                    "model": model,
                    "line": line_number,
                }
                annotated: dict[str, Any] = parsed
            except EXPECTED_ROW_ERRORS as error:
                elapsed = perf_counter() - row_started
                measured_seconds += elapsed
                failed += 1
                original = (
                    parsed
                    if isinstance(parsed, dict)
                    else {"source": line.rstrip("\n")}
                )
                original["covale"] = {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": elapsed,
                    "method": method,
                    "model": model,
                    "line": line_number,
                }
                annotated = original

            destination.write(json.dumps(annotated) + "\n")

    total_seconds = perf_counter() - started
    return {
        "input": str(input_path),
        "output": str(output_path),
        "method": method,
        "model": model,
        "rows": rows,
        "succeeded": succeeded,
        "failed": failed,
        "setup_seconds": setup_seconds,
        "total_seconds": total_seconds,
        "mean_row_seconds": measured_seconds / rows if rows else 0.0,
    }


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
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
