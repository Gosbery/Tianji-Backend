"""直接解读通道共用的提示词与文本工具行为。

旧的“检索 → 生成”路径在 Task 9 被移除，相关用例改由
`test_llm_direct.py` 覆盖；本文件只保留 direct 路径实际调用的部分。
"""

from __future__ import annotations

from datetime import date, time

from bazi_api.integrations.llm import AnswerGenerator, _humanize_internal_labels
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator


def chart():
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="隐私姓名")
    )


def test_chart_context_hides_internal_luck_status_values() -> None:
    context = AnswerGenerator._chart_context(chart())

    assert '"status"' not in context
    assert '"current"' not in context
    assert '"future"' not in context
    assert '"pattern_candidates"' not in context
    assert "候选" not in context


def test_direct_answers_drop_internal_status_labels() -> None:
    text = "当前辛亥（2018—2027，状态current），下一步壬子（2028—2037，status=future）。"

    assert _humanize_internal_labels(text) == "当前辛亥（2018—2027），下一步壬子（2028—2037）。"
