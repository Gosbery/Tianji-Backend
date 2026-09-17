from datetime import date, time

from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.charts.topic_facts import (
    annual_ganzhi,
    annual_pillars,
    element_distribution,
    peach_blossom_branches,
    tian_de_target,
    tian_yi_branches,
    wenchang_branch,
    yue_de_target,
)


def chart():
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )


def test_annual_ganzhi_follows_sixty_cycle() -> None:
    assert annual_ganzhi(1984) == "甲子"
    assert annual_ganzhi(1990) == "庚午"
    assert annual_ganzhi(2026) == "丙午"


def test_annual_pillars_covers_requested_window() -> None:
    pillars = annual_pillars(2026, 10)

    assert [item["year"] for item in pillars] == list(range(2026, 2036))
    assert pillars[0]["ganzhi"] == "丙午"
    assert pillars[9]["ganzhi"] == "乙卯"  # 2035：offset 51 → 乙卯


def test_element_distribution_counts_stems_and_hidden_stems() -> None:
    lines = element_distribution(chart().pillars)

    assert lines[0] == "火 5 处"
    assert "土 4 处" in lines
    assert "木 2 处" in lines
    assert "金 1 处" in lines
    assert "水 1 处" in lines


def test_spirit_star_lookup_tables() -> None:
    assert tian_yi_branches("甲") == ["丑", "未"]
    assert tian_yi_branches("丙") == ["亥", "酉"]
    assert wenchang_branch("丙") == "申"
    assert peach_blossom_branches("巳") == ["午"]
    assert peach_blossom_branches("寅") == ["卯"]
    assert tian_de_target("子") == "巳"
    assert yue_de_target("子") == "壬"
