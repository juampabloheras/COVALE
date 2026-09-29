import json
from pathlib import Path

from benchmark import run


def write_pairs(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"reference": "one", "candidate": "two"}),
                json.dumps({"reference": "three", "candidate": "four"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_annotate_jsonl_scores_every_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    write_pairs(input_path)
    monkeypatch.setattr(
        run,
        "evaluate",
        lambda *args, **kwargs: {
            "score": 0.75,
            "diagnostics": {"resolved": True},
        },
    )

    summary = run.annotate_jsonl(
        input_path,
        output_path,
        method="similarity",
        client=object(),
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["covale"]["score"] for row in rows] == [0.75, 0.75]
    assert all(row["covale"]["elapsed_seconds"] >= 0 for row in rows)
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["total_seconds"] >= summary["mean_row_seconds"]
    assert Path(summary["summary"]).is_file()
    assert len(summary["input_sha256"]) == 64
    assert len(summary["output_sha256"]) == 64


def test_annotate_jsonl_records_error_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    write_pairs(input_path)
    calls = 0

    def score(*args, **kwargs) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("unresolved")
        return {"score": 1.0, "diagnostics": {"resolved": True}}

    monkeypatch.setattr(run, "evaluate", score)

    summary = run.annotate_jsonl(input_path, output_path, client=object())

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["covale"]["status"] == "error"
    assert rows[0]["covale"]["error"] == "unresolved"
    assert rows[1]["covale"]["status"] == "ok"
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1


def test_annotate_jsonl_records_malformed_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    input_path.write_text(
        "not json\n"
        + json.dumps({"reference": "one", "candidate": "two"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run,
        "evaluate",
        lambda *args, **kwargs: {
            "score": 1.0,
            "diagnostics": {"resolved": True},
        },
    )

    summary = run.annotate_jsonl(input_path, output_path, client=object())

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["covale"]["error_type"] == "ValidationError"
    assert rows[1]["covale"]["score"] == 1.0
    assert summary["failed"] == 1


def test_annotate_jsonl_runs_rows_concurrently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from threading import Barrier

    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    write_pairs(input_path)
    barrier = Barrier(2)

    def score(*args, **kwargs) -> dict:
        barrier.wait(timeout=2)
        return {"score": 1.0, "diagnostics": {"resolved": True}}

    monkeypatch.setattr(run, "evaluate", score)

    summary = run.annotate_jsonl(
        input_path,
        output_path,
        concurrency=2,
        client=object(),
    )

    assert summary["concurrency"] == 2
    assert summary["succeeded"] == 2


def test_annotate_jsonl_supports_report_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    write_pairs(input_path)
    calls: list[bool] = []

    def score(*args, **kwargs) -> dict:
        calls.append(kwargs["extract_findings"])
        return {
            "score": 0.5,
            "diagnostics": {"resolved": True},
        }

    monkeypatch.setattr(run, "evaluate", score)

    summary = run.annotate_jsonl(
        input_path,
        output_path,
        extract_findings=True,
        client=object(),
    )

    assert calls == [True, True]
    assert summary["extract_findings"] is True
    assert len(summary["extraction_prompt_sha256"]) == 64
    assert len(summary["compatibility_prompt_sha256"]) == 64
