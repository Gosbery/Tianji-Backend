from datetime import date, datetime, time
from typing import Literal

import pytest
from pydantic import ValidationError

from bazi_api.modules.charts.schemas import BirthInput, PillarFacts
from bazi_api.modules.charts.service import ChartCalculator


def _stub_pillars(branches: list[str]) -> list[PillarFacts]:
    keys: list[Literal["year", "month", "day", "time"]] = ["year", "month", "day", "time"]
    labels = ["年柱", "月柱", "日柱", "时柱"]
    return [
        PillarFacts(
            key=keys[index],
            label=labels[index],
            stem="甲",
            branch=branch,
            stem_element="木",
            stem_yin_yang="阳",
            branch_element="木",
            hidden_stems=[],
            ten_god_stem="比肩",
            ten_god_branches=[],
            nayin="",
        )
        for index, branch in enumerate(branches)
    ]


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


@pytest.mark.parametrize(
    ("branches", "forbidden", "expected"),
    [
        # 隔角对（组内下标不相邻）不得再标半三合/半三会
        (["申", "辰", "申", "辰"], "半三合", None),
        (["亥", "丑", "亥", "丑"], "半三会", None),
        (["寅", "辰", "寅", "辰"], "半三会", None),
        (["巳", "未", "巳", "未"], "半三会", None),
        # 相邻对仍是半合/半会
        (["寅", "午", "寅", "午"], None, "寅午半三合候选"),
        (["亥", "子", "亥", "子"], None, "亥子半三会候选"),
        (["申", "子", "申", "子"], None, "申子半三合候选"),
        # 三支齐全仍标“全”
        (["申", "子", "辰", "午"], None, "申子辰全三合候选"),
    ],
)
def test_relations_only_mark_adjacent_pairs_as_partial_harmony(
    branches: list[str], forbidden: str | None, expected: str | None
) -> None:
    relations = ChartCalculator._relations(_stub_pillars(branches))
    labels = {item.label for item in relations}

    if forbidden is not None:
        assert not any(forbidden in label for label in labels), labels
    if expected is not None:
        assert expected in labels, labels


def test_chart_marks_time_boundary_uncertainty() -> None:
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(23, 30), name="测试")
    )
    assert any("换日" in item for item in chart.uncertainties)


def test_chart_calculates_current_and_next_luck_cycles_for_gender() -> None:
    male = ChartCalculator().calculate(
        BirthInput(date=date(2002, 9, 22), time=time(16, 0), gender="male")
    )
    female = ChartCalculator().calculate(
        BirthInput(date=date(2002, 9, 22), time=time(16, 0), gender="female")
    )

    assert male.luck.direction_label == "顺排"
    assert male.luck.start_at.isoformat() == "2008-02-07T10:00:00"
    assert [item.ganzhi for item in male.luck.cycles[:3]] == ["庚戌", "辛亥", "壬子"]
    assert female.luck.direction_label == "逆排"
    assert female.luck.start_at.isoformat() == "2007-07-25T02:00:00"
    assert [item.ganzhi for item in female.luck.cycles[:3]] == ["戊申", "丁未", "丙午"]
    for chart in (male, female):
        expected_current = next(
            (
                item
                for item in chart.luck.cycles
                if item.start_year <= datetime.now().year <= item.end_year
            ),
            None,
        )
        assert chart.luck.current_cycle == expected_current
        if expected_current:
            assert chart.luck.next_cycle
            assert chart.luck.next_cycle.index == expected_current.index + 1


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
