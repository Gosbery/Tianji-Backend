from pathlib import Path

import pytest

from bazi_api.cli.evaluate import (
    EvaluationConfigurationError,
    compare_with_baseline,
    parse_parameter_grid,
    run,
    settings_with_tuning,
)
from bazi_api.core.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        knowledge_path=tmp_path / "knowledge",
        evals_path=tmp_path / "evals",
        database_path=tmp_path / "app.db",
    )


def _baseline_identity() -> dict[str, object]:
    return {
        "mode": "bm25",
        "dataset": "questions.json",
        "dataset_sha256": "dataset-digest",
        "cases": 60,
        "limit": 10,
        "evidence_scope": "reviewed_only",
        "answer_mode": "off",
        "model_version": "bm25-lexicon-v1",
        "index_version": "index-v1",
        "tuning": {"bm25_k1": 1.2},
        "max_regression": 0.02,
    }


def test_parameter_grid_validates_and_expands_combinations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    grid = parse_parameter_grid(["bm25_k1=1.0,1.2", "concept_boost_max=0,0.1"], settings)

    assert len(grid) == 4
    assert {item["bm25_k1"] for item in grid} == {1.0, 1.2}
    assert settings_with_tuning(settings, grid[-1]).concept_boost_max == 0.1
    with pytest.raises(EvaluationConfigurationError, match="不支持"):
        parse_parameter_grid(["unknown=1"], settings)
    with pytest.raises(EvaluationConfigurationError):
        settings_with_tuning(settings, {"bm25_k1": 0})
    with pytest.raises(EvaluationConfigurationError):
        parse_parameter_grid(["sparse_recall_limit=-1"], settings)


def test_baseline_comparison_fails_on_regression() -> None:
    baseline = {
        **_baseline_identity(),
        "recall_at_5": 0.95,
        "mrr_at_10": 0.90,
    }
    report = {**baseline, "recall_at_5": 0.89, "mrr_at_10": 0.89}

    comparison = compare_with_baseline(report, baseline, max_regression=0.02)

    assert not comparison["passed"]
    assert not comparison["metrics"]["recall_at_5"]["passed"]
    assert comparison["metrics"]["mrr_at_10"]["passed"]


def test_baseline_metrics_missing_from_current_report_are_rejected() -> None:
    baseline = {**_baseline_identity(), "recall_at_5": 1.0, "citation_chain_rate": 1.0}
    report = {**_baseline_identity(), "recall_at_5": 1.0}

    with pytest.raises(EvaluationConfigurationError, match="citation_chain_rate"):
        compare_with_baseline(report, baseline)


def test_baseline_binds_dataset_index_tuning_and_regression_policy() -> None:
    baseline = {**_baseline_identity(), "recall_at_5": 1.0}

    for field, changed in (
        ("dataset_sha256", "different-dataset"),
        ("cases", 1),
        ("limit", 20),
        ("answer_mode", "offline"),
        ("model_version", "different-model"),
        ("index_version", "different-index"),
        ("tuning", {"bm25_k1": 1.0}),
    ):
        report = {**baseline, field: changed}
        with pytest.raises(EvaluationConfigurationError, match=field):
            compare_with_baseline(report, baseline)

    with pytest.raises(EvaluationConfigurationError, match="超过基线允许值"):
        compare_with_baseline(baseline, baseline, max_regression=0.03)
    with pytest.raises(EvaluationConfigurationError, match="有限数值"):
        compare_with_baseline(baseline, baseline, max_regression=float("inf"))


@pytest.mark.asyncio
async def test_evaluation_rejects_limit_too_small(tmp_path: Path) -> None:
    with pytest.raises(EvaluationConfigurationError, match="至少为 10"):
        await run("bm25", 1, settings=_settings(tmp_path))
