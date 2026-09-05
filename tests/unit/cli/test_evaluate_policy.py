import pytest

from bazi_api.cli.evaluate_policy import evaluate_policy_cases


def test_policy_evaluation_requires_exact_classification() -> None:
    report = evaluate_policy_cases(
        [
            {"id": "wealth", "question": "我何年暴富？", "expected_policy": "evidence_answer"},
            {
                "id": "mapping",
                "question": "金是否可以直接解释成财富？",
                "expected_policy": "evidence_answer",
            },
            {
                "id": "research",
                "question": "原文如何定义财星？",
                "expected_policy": "evidence_answer",
            },
        ]
    )

    assert report["policy_accuracy_rate"] == 1.0
    assert report["passed"] is True


def test_policy_evaluation_rejects_incomplete_cases() -> None:
    with pytest.raises(ValueError, match="缺少必填字段"):
        evaluate_policy_cases([{"id": "missing"}])
