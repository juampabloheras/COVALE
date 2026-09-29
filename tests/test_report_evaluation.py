import json
from importlib import import_module
from pathlib import Path

import nibabel as nib
import numpy as np

from covale import COVALE
from covale.evaluator import align_units, evaluate
from covale.evaluator.report_processing import AnatomicalUnit


class FakeResponses:
    def __init__(self, *outputs: dict[str, object]) -> None:
        self.outputs = iter(json.dumps(output) for output in outputs)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return type("Response", (), {"output_text": next(self.outputs)})()


class FakeClient:
    def __init__(self, *outputs: dict[str, object]) -> None:
        self.responses = FakeResponses(*outputs)


def finding(
    text: str,
    anatomy: str,
    *,
    concept: str = "lesion",
    assertion: str = "present",
) -> dict[str, str]:
    return {
        "text": text,
        "anatomy": anatomy,
        "concept": concept,
        "assertion": assertion,
    }


def extraction(*findings: dict[str, str]) -> dict[str, object]:
    return {"findings": list(findings)}


def compatibility(*pairs: tuple[int, int]) -> dict[str, object]:
    return {
        "compatible_pairs": [
            {
                "reference_index": reference,
                "candidate_index": candidate,
            }
            for reference, candidate in pairs
        ]
    }


def registry(tmp_path: Path) -> Path:
    masks = tmp_path / "masks"
    masks.mkdir()
    affine = np.eye(4)
    nib.save(
        nib.Nifti1Image(np.array([1, 1, 0], dtype=np.uint8), affine),
        masks / "a.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(np.array([0, 1, 1], dtype=np.uint8), affine),
        masks / "b.nii.gz",
    )
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "atlas": "test",
                "space": "test",
                "regions": {
                    "region a": {
                        "mask": "masks/a.nii.gz",
                        "aliases": [],
                    },
                    "region b": {
                        "mask": "masks/b.nii.gz",
                        "aliases": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def expression(text: str, _registry: object) -> dict[str, str]:
    return {"op": "region", "name": text}


def test_alignment_maximizes_total_one_to_one_score() -> None:
    references = [
        AnatomicalUnit(text="r0", anatomy="r0"),
        AnatomicalUnit(text="r1", anatomy="r1"),
    ]
    candidates = [
        AnatomicalUnit(text="c0", anatomy="c0"),
        AnatomicalUnit(text="c1", anatomy="c1"),
    ]
    scores = {
        ("r0", "c0"): 0.1,
        ("r0", "c1"): 0.9,
        ("r1", "c0"): 0.8,
        ("r1", "c1"): 0.2,
    }

    alignment = align_units(
        references,
        candidates,
        compatible={(0, 0), (0, 1), (1, 0), (1, 1)},
        score_pair=lambda reference, candidate: {
            "score": scores[(reference.anatomy, candidate.anatomy)],
            "diagnostics": {"resolved": True},
        },
    )

    assert {
        (match.reference_index, match.candidate_index) for match in alignment.matches
    } == {(0, 1), (1, 0)}
    assert alignment.missing == []
    assert alignment.spurious == []


def test_report_evaluation_extracts_aligns_and_scores(tmp_path: Path) -> None:
    client = FakeClient(
        extraction(
            finding("first reference", "region a"),
            finding("second reference", "region b"),
        ),
        extraction(
            finding("first candidate", "region b"),
            finding("second candidate", "region a"),
        ),
        compatibility((0, 0), (0, 1), (1, 0), (1, 1)),
    )

    result = evaluate(
        "reference report",
        "candidate report",
        registry(tmp_path),
        expression,
        extract_findings=True,
        client=client,
    )

    assert result["score"] == 1.0
    assert {
        (match["reference_index"], match["candidate_index"])
        for match in result["diagnostics"]["matches"]
    } == {(0, 1), (1, 0)}
    assert result["diagnostics"]["missing"] == []
    assert result["diagnostics"]["spurious"] == []
    assert len(client.responses.requests) == 3


def test_report_evaluation_penalizes_missing_findings(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        extraction(
            finding("first reference", "region a"),
            finding("second reference", "region b"),
        ),
        extraction(finding("candidate", "region a")),
        compatibility((0, 0)),
    )

    result = evaluate(
        "reference report",
        "candidate report",
        registry(tmp_path),
        expression,
        extract_findings=True,
        client=client,
    )

    assert result["score"] == 0.5
    assert [item["anatomy"] for item in result["diagnostics"]["missing"]] == [
        "region b"
    ]
    assert result["diagnostics"]["spurious"] == []


def test_report_evaluation_rejects_assertion_mismatch(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        extraction(finding("reference", "region a")),
        extraction(finding("candidate", "region a", assertion="absent")),
        compatibility((0, 0)),
    )

    result = evaluate(
        "reference report",
        "candidate report",
        registry(tmp_path),
        expression,
        extract_findings=True,
        client=client,
    )

    assert result["score"] == 0.0
    assert result["diagnostics"]["matches"] == []
    assert len(result["diagnostics"]["missing"]) == 1
    assert len(result["diagnostics"]["spurious"]) == 1


def test_report_evaluation_scores_two_empty_reports_as_equal(
    tmp_path: Path,
) -> None:
    client = FakeClient(extraction(), extraction())

    result = evaluate(
        "normal reference report",
        "normal candidate report",
        registry(tmp_path),
        expression,
        extract_findings=True,
        client=client,
    )

    assert result["score"] == 1.0
    assert result["diagnostics"]["matches"] == []
    assert len(client.responses.requests) == 2


def test_report_evaluation_reuses_created_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = FakeClient(
        extraction(finding("reference", "region a")),
        extraction(finding("candidate", "region a")),
        compatibility((0, 0)),
        {"op": "region", "name": "region a"},
        {"op": "region", "name": "region a"},
    )
    created = 0

    def create_client() -> FakeClient:
        nonlocal created
        created += 1
        return client

    evaluate_module = import_module("covale.evaluator.evaluate")
    monkeypatch.setattr(evaluate_module, "create_client", create_client)

    result = evaluate(
        "reference report",
        "candidate report",
        registry(tmp_path),
        method="llm",
        extract_findings=True,
    )

    assert result["score"] == 1.0
    assert created == 1
    assert len(client.responses.requests) == 3


def test_evaluator_forwards_extract_findings(monkeypatch) -> None:
    arguments: dict[str, object] = {}

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        arguments.update(kwargs)
        return {
            "score": 1.0,
            "diagnostics": {"resolved": True},
        }

    monkeypatch.setattr(
        "covale.evaluator.core.evaluate_pair",
        fake_evaluate,
    )
    evaluator = COVALE(
        method="similarity",
        extract_findings=True,
        client=object(),
    )

    assert evaluator.score("reference", "candidate") == 1.0
    assert arguments["extract_findings"] is True


def test_config_enables_finding_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
metrics: [dice]
extract_findings: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "covale.evaluator.core.create_client",
        lambda: object(),
    )

    evaluator = COVALE.from_config(config)

    assert evaluator.extract_findings is True
