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
