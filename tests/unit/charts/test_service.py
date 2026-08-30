from datetime import date, time

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
