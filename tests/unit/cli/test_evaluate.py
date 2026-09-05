from pathlib import Path

import pytest

from bazi_api.cli.evaluate import (
    EvaluationConfigurationError,
    compare_with_baseline,
    evaluate_generated_answer,
    parse_parameter_grid,
    run,
    settings_with_tuning,
)
from bazi_api.core.config import Settings
from bazi_api.integrations.llm import GenerationResult
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        knowledge_path=tmp_path / "knowledge",
        evals_path=tmp_path / "evals",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
        embedding_cache_path=tmp_path / "embeddings.sqlite3",
        embedding_provider="hash",
        reranker_provider="lexical",
    )


def _hit() -> RetrievalHit:
    document = RetrievalDocument(
        id="evidence-1",
        kind="knowledge_card",
        layer=3,
        title="证据",
        text="证据内容",
        source="test",
        school="基础共识",
        concepts=[],
    )
    return RetrievalHit(document=document, score=1.0, matched_by=["test"])


def _baseline_identity() -> dict[str, object]:
    return {
        "mode": "hybrid",
        "dataset": "questions.json",
        "dataset_sha256": "dataset-digest",
        "cases": 60,
        "limit": 10,
        "evidence_scope": "reviewed_only",
        "answer_mode": "offline",
        "embedding_provider": "hash",
        "reranker_provider": "lexical",
        "model_version": "hash-v1",
        "index_version": "index-v1",
        "tuning": {"retrieval_rrf_k": 60},
        "max_regression": 0.02,
    }


def test_parameter_grid_validates_and_expands_combinations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    grid = parse_parameter_grid(
        ["retrieval_rrf_k=30,60", "concept_boost_max=0,0.1"], settings
    )

    assert len(grid) == 4
    assert {item["retrieval_rrf_k"] for item in grid} == {30, 60}
    assert settings_with_tuning(settings, grid[-1]).concept_boost_max == 0.1
    with pytest.raises(EvaluationConfigurationError, match="不支持"):
        parse_parameter_grid(["unknown=1"], settings)
    with pytest.raises(EvaluationConfigurationError):
        settings_with_tuning(settings, {"retrieval_rrf_k": 0})
    with pytest.raises(EvaluationConfigurationError):
        parse_parameter_grid(["dense_recall_limit=-1"], settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_provider", "sentence-transformers"),
        ("vector_backend", "qdrant-local"),
        ("reranker_provider", "cross-encoder"),
    ],
)
def test_provider_typos_fail_configuration(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(ValueError):
        Settings.model_validate({**_settings(tmp_path).model_dump(), field: value})


def test_answer_quality_checks_citations_and_allows_prediction_output() -> None:
    invalid = evaluate_generated_answer(
        GenerationResult(
            answer="[2] 可以断定结果。",
            uncertainties=[],
            followups=[],
            citations_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=True,
    )
    valid = evaluate_generated_answer(
        GenerationResult(
            answer="明年事业会有进展 [1]。",
            uncertainties=["具体结果仍受现实条件影响"],
            followups=[],
            citations_validated=True,
            uncertainty_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=True,
    )
    superficial = evaluate_generated_answer(
        GenerationResult(
            answer="不能 [1]",
            uncertainties=["  "],
            followups=[],
            citations_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=True,
    )
    reversal = evaluate_generated_answer(
        GenerationResult(
            answer="并不是不能依据证据断定，而是一定会成功 [1]。",
            uncertainties=["仍需关注"],
            followups=[],
            citations_validated=True,
            uncertainty_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=True,
    )
    contradictory = evaluate_generated_answer(
        GenerationResult(
            answer="不能依据证据断定。你必会成功 [1]。",
            uncertainties=["毫无不确定性，已经确定会成功"],
            followups=[],
            citations_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=True,
    )

    assert invalid == {
        "citation_valid": False,
        "policy_correct": True,
        "answer_allowed": True,
        "uncertainty_triggered": False,
    }
    assert all(valid.values())
    assert superficial == {
        "citation_valid": True,
        "policy_correct": True,
        "answer_allowed": True,
        "uncertainty_triggered": False,
    }
    assert reversal == {
        "citation_valid": True,
        "policy_correct": True,
        "answer_allowed": True,
        "uncertainty_triggered": True,
    }
    assert contradictory == {
        "citation_valid": True,
        "policy_correct": True,
        "answer_allowed": True,
        "uncertainty_triggered": False,
    }

    refused = evaluate_generated_answer(
        GenerationResult(
            answer="不能作答 [1]。",
            uncertainties=["边界"],
            followups=[],
            policy_decision="refuse_no_evidence",
            citations_validated=True,
            uncertainty_validated=True,
        ),
        [_hit()],
        expected_policy="evidence_answer",
        expects_uncertainty=False,
    )
    assert refused["policy_correct"]
    assert not refused["answer_allowed"]


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


def test_baseline_answer_metrics_cannot_be_bypassed() -> None:
    baseline = {
        **_baseline_identity(),
        "answer_quality": {"citation_validity_rate": 1.0},
    }
    without_answers = {
        **_baseline_identity(),
        "answer_mode": "off",
    }

    with pytest.raises(EvaluationConfigurationError, match="answer_mode"):
        compare_with_baseline(without_answers, baseline)

    without_answers["answer_mode"] = "offline"
    with pytest.raises(EvaluationConfigurationError, match="citation_validity_rate"):
        compare_with_baseline(without_answers, baseline)


def test_baseline_binds_dataset_index_tuning_and_regression_policy() -> None:
    baseline = {**_baseline_identity(), "recall_at_5": 1.0}

    for field, changed in (
        ("dataset_sha256", "different-dataset"),
        ("cases", 1),
        ("limit", 20),
        ("model_version", "different-model"),
        ("index_version", "different-index"),
        ("tuning", {"retrieval_rrf_k": 30}),
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
        await run("hybrid", 1, settings=_settings(tmp_path))


@pytest.mark.asyncio
async def test_lightrag_evaluation_fails_cleanly_when_unconfigured(tmp_path: Path) -> None:
    with pytest.raises(EvaluationConfigurationError, match="LightRAG 未配置"):
        await run("lightrag", 10, settings=_settings(tmp_path))
