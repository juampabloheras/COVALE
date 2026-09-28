from covale import COVALE, compare_systems


def test_covale_callable_returns_corpus_mean(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def score(reference: str, candidate: str, *args, **kwargs) -> float:
        calls.append((reference, candidate))
        return 1.0 if reference == candidate else 0.5

    monkeypatch.setattr("covale.evaluator.score_covale", score)
    monkeypatch.setattr("covale.evaluator.create_client", object)
    evaluator = COVALE(method="similarity")

    first = evaluator(
        refs=["same", "reference"],
        hyps=["same", "candidate"],
    )
    second = evaluator(
        refs=["same", "reference"],
        hyps=["same", "candidate"],
    )

    assert first == {"dice": 0.75}
    assert second == first
    assert calls == [("same", "same"), ("reference", "candidate")]


def test_covale_callable_can_return_per_sample_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        "covale.evaluator.score_covale",
        lambda *args, **kwargs: 0.25,
    )
    monkeypatch.setattr("covale.evaluator.create_client", object)
    evaluator = COVALE(method="similarity", per_sample=True)

    assert evaluator(["a", "b"], ["c", "d"]) == {
        "dice": [0.25, 0.25]
    }


def test_covale_from_config(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "atlas_registry" / "registry.json"
    registry.parent.mkdir()
    registry.write_text('{"regions": {}}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        """
metrics:
  - dice:
      method: similarity
      provider: openai
      model_name: test-model
      registry_path: atlas_registry/registry.json
output:
  mode: detailed
cache: false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "covale.evaluator.score_covale",
        lambda *args, **kwargs: 0.5,
    )
    monkeypatch.setattr("covale.evaluator.create_client", object)

    evaluator = COVALE.from_config(config)
    result = evaluator(["a", "b"], ["c", "d"])

    assert evaluator.method == "similarity"
    assert evaluator.model == "test-model"
    assert evaluator.registry_path == registry
    assert evaluator.cache is False
    assert result == {
        "dice": 0.5,
        "dice_std": 0.0,
        "dice_per_sample": [0.5, 0.5],
    }


def test_covale_from_config_rejects_unknown_metric(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("metrics: [bleu]", encoding="utf-8")

    try:
        COVALE.from_config(config)
    except ValueError as error:
        assert "dice" in str(error)
    else:
        raise AssertionError("Expected an unknown metric to fail.")


def test_compare_systems_caches_pair_scores_before_bootstrap() -> None:
    calls = 0

    def accuracy(hyps, refs) -> float:
        nonlocal calls
        calls += 1
        return float(hyps[0] == refs[0])

    signatures, scores = compare_systems(
        systems={
            "baseline": ["wrong", "b", "wrong"],
            "improved": ["a", "b", "c"],
        },
        metrics={"dice": accuracy},
        references=["a", "b", "c"],
        n_samples=1_000,
        random_seed=7,
    )

    assert calls == 6
    assert scores["baseline"]["dice"] == 1 / 3
    assert scores["improved"]["dice"] == 1.0
    assert 0 <= scores["improved"]["dice_pvalue"] <= 1
    assert signatures[0]["baseline"] == "baseline"
    assert signatures[0]["system"] == "improved"
    assert signatures[0]["delta"] == 2 / 3
    assert signatures[0]["n_samples"] == 1_000


def test_compare_systems_validates_lengths() -> None:
    try:
        compare_systems(
            systems={"baseline": ["a"], "improved": ["a", "b"]},
            metrics={"dice": lambda hyps, refs: 1.0},
            references=["a"],
        )
    except ValueError as error:
        assert "improved" in str(error)
    else:
        raise AssertionError("Expected mismatched system lengths to fail.")
