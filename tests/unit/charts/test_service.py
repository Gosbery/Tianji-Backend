from datetime import date, time

import pytest
from pydantic import ValidationError

from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator


def test_chart_is_deterministic() -> None:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )

    assert [f"{pillar.stem}{pillar.branch}" for pillar in chart.pillars] == [
        "己巳",
        "丙子",
        "丙寅",
        "甲午",
    ]
    assert chart.day_master == "丙"
    assert chart.day_master_element == "火"
    assert "大模型" in chart.calculation_basis


def test_chart_includes_structural_facts_without_claiming_transformation() -> None:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )

    assert chart.month_command.model_dump() == {
        "branch": "子",
        "main_hidden_stem": "癸",
        "ten_god": "正官",
    }
    assert chart.pattern_candidates[0].name == "正官格候选"
    assert chart.pattern_candidates[0].basis == "month_main_qi"
    assert chart.pattern_candidates[0].exposed_positions == []
    assert "仅为候选" in chart.pattern_candidates[0].note
    assert any(
        item.stem_position == "day" and item.branch in {"巳", "寅"} and item.relation == "same_stem"
        for item in chart.roots
    )
    assert any(
        item.stem_position == "day" and item.branch == "午" and item.relation == "same_element"
        for item in chart.roots
    )
    assert {item.label for item in chart.structural_relations} >= {
        "子午冲",
        "寅午半三合候选",
        "寅巳刑候选",
    }
    assert any(item.hidden_stem == "丙" and item.outside_day_master for item in chart.exposed_stems)


def test_chart_marks_time_boundary_uncertainty() -> None:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(23, 30), name="测试")
    )
    assert any("换日" in item for item in chart.uncertainties)


def test_only_algorithm_supported_timezone_is_accepted() -> None:
    shanghai = ChartCalculator().calculate(
        BirthInput(
            date=date(1990, 1, 1),
            time=time(12, 0),
            timezone="Asia/Shanghai",
            name="测试",
        )
    )
    assert "Asia/Shanghai 当地民用时间" in shanghai.calculation_basis

    with pytest.raises(ValidationError, match="仅支持 Asia/Shanghai"):
        BirthInput(
            date=date(1990, 1, 1),
            time=time(12, 0),
            timezone="Asia/Tokyo",
        )
