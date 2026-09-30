"""Compare COVALE and RadEval metrics across repeated anatomical queries."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from covale import COVALE
from radeval import RadEval
from tqdm.auto import tqdm

REFS = [
    "hippocampal formation",
    "striatum",
    "striatum",
    "basal ganglia",
    "basal ganglia",
    "basal ganglia",
    "diencephalon",
    "diencephalon",
    "medial temporal lobe",
    "medial temporal lobe",
    "medial temporal lobe",
    "frontal lobe",
    "parietal lobe",
    "occipital lobe",
    "cerebral cortex",
    "parahippocampal gyrus",
    "amygdala",
    "lateral ventricle",
    "third ventricle",
    "external capsule",
    "internal capsule",
    "postcentral gyrus",
    "posterior cingulate cortex",
    "precuneus",
    "parahippocampal cortex",
    "cerebellum",
    "occipital cortex",
    "brainstem",
    "frontal pole",
    "primary visual cortex",
]

HYPS = [
    "hippocampus",
    "caudate",
    "putamen",
    "caudate",
    "putamen",
    "globus pallidus",
    "thalamus",
    "hypothalamus",
    "amygdala",
    "hippocampus",
    "parahippocampal gyrus",
    "precentral gyrus",
    "postcentral gyrus",
    "calcarine cortex",
    "insula",
    "hippocampus",
    "hippocampus",
    "caudate",
    "thalamus",
    "putamen",
    "globus pallidus",
    "precentral gyrus",
    "precuneus",
    "cuneus",
    "entorhinal cortex",
    "hippocampus",
    "caudate",
    "putamen",
    "thalamus",
    "amygdala",
]

COVALE_METHODS = ("llm", "deep_agent", "similarity")
RADEVAL_METRICS = (
    "bleu",
    "rouge",
    "bertscore",
    "radeval_bertscore",
)
RADEVAL_OUTPUT_KEYS = {
    "bleu": ("bleu",),
    "rouge": ("rouge1", "rouge2", "rougeL"),
    "bertscore": ("bertscore",),
    "radeval_bertscore": ("radeval_bertscore",),
}
QUICK_PAIR_INDICES = (0, 1, 11, 17, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark COVALE and RadEval metrics across repeated calls."
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
    )
    parser.add_argument(
        "--covale-methods",
        nargs="+",
        choices=COVALE_METHODS,
        default=list(COVALE_METHODS),
    )
    parser.add_argument("--skip-covale", action="store_true")
    parser.add_argument("--skip-radeval", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one repetition of the first pair.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run five representative pairs for three repetitions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing raw results instead of resuming them.",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.smoke and args.quick:
        parser.error("--smoke and --quick cannot be used together")
    return args


def result_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["framework"]),
        str(row["metric"]),
        int(row["pair_index"]),
        int(row["repetition"]),
    )


def load_completed(path: Path) -> set[tuple[str, str, int, int]]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add(result_key(row))
    return completed


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def load_latest_rows(path: Path) -> list[dict[str, Any]]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[result_key(row)] = row
    return list(latest.values())


def base_row(
    framework: str,
    metric: str,
    pair_index: int,
    repetition: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "metric": metric,
        "pair_index": pair_index,
        "repetition": repetition,
        "reference": REFS[pair_index],
        "hypothesis": HYPS[pair_index],
        "elapsed_seconds": elapsed_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_covale(
    method: str,
    pair_index: int,
    repetition: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    row = base_row("covale", f"covale_{method}", pair_index, repetition, 0.0)
    try:
        # A new evaluator and disabled cache guarantee an independent model call.
        evaluator = COVALE(method=method, cache=False)
        result = evaluator.evaluate_pair(
            REFS[pair_index],
            HYPS[pair_index],
        )
        row.update(
            {
                "status": "ok",
                "score": float(result["score"]),
                "details": {
                    key: result[key]
                    for key in (
                        "reference_expression",
                        "candidate_expression",
                        "diagnostics",
                    )
                    if key in result
                },
            }
        )
    except Exception as error:  # noqa: BLE001 - benchmark records each failure
        row.update(
            {
                "status": "error",
                "score": None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def run_radeval_metric(
    evaluator: RadEval,
    output_keys: tuple[str, ...],
    refs: list[str],
    hyps: list[str],
    pair_indices: list[int],
    repetition: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    try:
        scores = evaluator(refs=refs, hyps=hyps)
    except Exception as error:  # noqa: BLE001 - benchmark records each failure
        elapsed = time.perf_counter() - started
        per_pair_elapsed = elapsed / len(refs)
        return [
            {
                **base_row(
                    "radeval",
                    metric,
                    pair_indices[pair_index],
                    repetition,
                    per_pair_elapsed,
                ),
                "status": "error",
                "score": None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            for metric in output_keys
            for pair_index in range(len(refs))
        ]

    elapsed = time.perf_counter() - started
    per_pair_elapsed = elapsed / len(refs)
    rows = []
    for metric, values in scores.items():
        for local_index, score in enumerate(values):
            rows.append(
                {
                    **base_row(
                        "radeval",
                        metric,
                        pair_indices[local_index],
                        repetition,
                        per_pair_elapsed,
                    ),
                    "status": "ok",
                    "score": float(score),
                }
            )
    return rows


def summarize(raw_path: Path, output_dir: Path) -> None:
    rows = load_latest_rows(raw_path)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["framework"]),
            str(row["metric"]),
            int(row["pair_index"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (framework, metric, pair_index), group in sorted(grouped.items()):
        scores = [
            float(row["score"])
            for row in group
            if row["status"] == "ok" and row["score"] is not None
        ]
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        summary.append(
            {
                "framework": framework,
                "metric": metric,
                "pair_index": pair_index,
                "reference": REFS[pair_index],
                "hypothesis": HYPS[pair_index],
                "attempted": len(group),
                "succeeded": len(scores),
                "failed": len(group) - len(scores),
                "score_mean": statistics.fmean(scores) if scores else None,
                "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "elapsed_mean_seconds": statistics.fmean(elapsed),
            }
        )

    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if summary:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

    metric_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["framework"]), str(row["metric"]))
        metric_groups.setdefault(key, []).append(row)

    metric_summary = []
    for (framework, metric), group in sorted(metric_groups.items()):
        successful = [
            row
            for row in group
            if row["status"] == "ok" and row["score"] is not None
        ]
        scores = [float(row["score"]) for row in successful]
        by_repetition: dict[int, list[float]] = {}
        for row in successful:
            by_repetition.setdefault(int(row["repetition"]), []).append(
                float(row["score"])
            )
        run_means = [
            statistics.fmean(run_scores)
            for _, run_scores in sorted(by_repetition.items())
        ]
        by_pair: dict[int, list[float]] = {}
        for row in successful:
            by_pair.setdefault(int(row["pair_index"]), []).append(
                float(row["score"])
            )
        pair_variances = [
            statistics.variance(pair_scores)
            if len(pair_scores) > 1
            else 0.0
            for pair_scores in by_pair.values()
        ]
        elapsed = [
            float(row["elapsed_seconds"])
            for row in successful
        ]
        metric_summary.append(
            {
                "framework": framework,
                "metric": metric,
                "attempted": len(group),
                "succeeded": len(scores),
                "failed": len(group) - len(scores),
                "all_scores_mean": (
                    statistics.fmean(scores) if scores else None
                ),
                "all_scores_std": (
                    statistics.stdev(scores) if len(scores) > 1 else 0.0
                ),
                "run_mean": (
                    statistics.fmean(run_means) if run_means else None
                ),
                "run_mean_std": (
                    statistics.stdev(run_means)
                    if len(run_means) > 1
                    else 0.0
                ),
                "run_count": len(run_means),
                "mean_pair_variance": (
                    statistics.fmean(pair_variances)
                    if pair_variances
                    else None
                ),
                "mean_completion_seconds": (
                    statistics.fmean(elapsed) if elapsed else None
                ),
            }
        )

    metric_json_path = output_dir / "metric_summary.json"
    metric_csv_path = output_dir / "metric_summary.csv"
    metric_json_path.write_text(
        json.dumps(metric_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if metric_summary:
        with metric_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(metric_summary[0]),
            )
            writer.writeheader()
            writer.writerows(metric_summary)


def main() -> None:
    args = parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not args.skip_covale and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Add OPENAI_API_KEY to the repository-root .env before running "
            "COVALE benchmark methods."
        )
    if args.smoke:
        pair_indices = [0]
        repetitions = 1
    elif args.quick:
        pair_indices = list(QUICK_PAIR_INDICES)
        repetitions = 3
    else:
        pair_indices = list(range(len(REFS)))
        repetitions = args.repetitions
    refs = [REFS[index] for index in pair_indices]
    hyps = [HYPS[index] for index in pair_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"
    if args.overwrite and raw_path.exists():
        raw_path.unlink()
    raw_path.touch(exist_ok=True)
    completed = load_completed(raw_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "covale_version": importlib.metadata.version("covale"),
        "radeval_version": importlib.metadata.version("radeval"),
        "references": refs,
        "hypotheses": hyps,
        "repetitions": repetitions,
        "covale_methods": args.covale_methods,
        "radeval_metrics": list(RADEVAL_METRICS),
        "covale_cache": False,
        "covale_evaluator_scope": "fresh evaluator per pair and repetition",
        "radeval_call_scope": "fresh metric call per repetition",
        "workers": args.workers,
        "smoke": args.smoke,
        "quick": args.quick,
        "pair_indices": pair_indices,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if not args.skip_covale:
        jobs = [
            (method, pair_index, repetition)
            for method in args.covale_methods
            for pair_index in pair_indices
            for repetition in range(1, repetitions + 1)
            if ("covale", f"covale_{method}", pair_index, repetition)
            not in completed
        ]
        print(f"Running {len(jobs)} independent COVALE evaluations...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_covale, *job): job
                for job in jobs
            }
            with tqdm(
                total=len(futures),
                desc="COVALE",
                unit="evaluation",
            ) as progress:
                for future in as_completed(futures):
                    row = future.result()
                    append_rows(raw_path, [row])
                    progress.set_postfix(
                        metric=row["metric"],
                        pair=row["pair_index"],
                        run=row["repetition"],
                        status=row["status"],
                    )
                    progress.update()

    if not args.skip_radeval:
        print("Loading RadEval metrics...")
        evaluators = {
            metric: RadEval(
                metrics=[metric],
                per_sample=True,
                show_progress=False,
            )
            for metric in RADEVAL_METRICS
        }
        with tqdm(
            total=repetitions * len(evaluators),
            desc="RadEval",
            unit="metric run",
        ) as progress:
            for repetition in range(1, repetitions + 1):
                for metric, evaluator in evaluators.items():
                    output_keys = RADEVAL_OUTPUT_KEYS[metric]
                    missing = any(
                        ("radeval", output_key, pair_index, repetition)
                        not in completed
                        for output_key in output_keys
                        for pair_index in pair_indices
                    )
                    if missing:
                        rows = run_radeval_metric(
                            evaluator,
                            output_keys,
                            refs,
                            hyps,
                            pair_indices,
                            repetition,
                        )
                        rows = [
                            row for row in rows
                            if result_key(row) not in completed
                        ]
                        append_rows(raw_path, rows)
                    progress.set_postfix(
                        metric=metric,
                        run=repetition,
                    )
                    progress.update()

    summarize(raw_path, args.output_dir)
    print(f"Raw results: {raw_path}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")
    print(f"Summary JSON: {args.output_dir / 'summary.json'}")
    print(f"Metric summary: {args.output_dir / 'metric_summary.csv'}")
    failures = sum(
        row.get("status") == "error"
        for row in load_latest_rows(raw_path)
    )
    if failures:
        raise SystemExit(
            f"Benchmark completed with {failures} failed rows. "
            "Correct the errors and rerun without --overwrite to retry them."
        )


if __name__ == "__main__":
    main()
