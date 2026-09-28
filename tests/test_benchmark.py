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
    monkeypatch.setattr(run, "covale", lambda *args, **kwargs: 0.75)

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


def test_annotate_jsonl_records_error_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "annotated.jsonl"
    write_pairs(input_path)
    calls = 0

    def score(*args, **kwargs) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("unresolved")
        return 1.0

    monkeypatch.setattr(run, "covale", score)

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
    monkeypatch.setattr(run, "covale", lambda *args, **kwargs: 1.0)

    summary = run.annotate_jsonl(input_path, output_path, client=object())

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["covale"]["error_type"] == "JSONDecodeError"
    assert rows[1]["covale"]["score"] == 1.0
    assert summary["failed"] == 1
