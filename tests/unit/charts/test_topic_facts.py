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


def test_wealth_pack_reports_star_positions_and_counts() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:wealth", current_year=2026)

    assert pack is not None
    assert pack.topic_id == "topic:wealth"
    joined = "\n".join(pack.facts)
    assert "偏财（庚金）藏于年支巳" in joined
    assert "正财（辛金）四柱未见" in joined
    assert "仅为计数对比" in joined
    assert any("流年" in fact and "2026" in fact for fact in pack.facts)


def test_career_pack_reports_officer_and_seal_stars() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:career", current_year=2026)

    joined = "\n".join(pack.facts)
    assert "正官（癸水）藏于月支子" in joined
    assert "七杀（壬水）四柱未见" in joined
    assert "偏印（甲木）透于时柱" in joined


def test_relationship_pack_reports_spouse_palace_by_gender() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:relationships", current_year=2026)

    joined = "\n".join(pack.facts)
    assert "配偶宫为日支寅（藏干甲、丙、戊）" in joined
    assert "寅午半三合候选" in joined or "寅巳刑候选" in joined
    assert "男命看财星" in joined and "女命看官杀" in joined


def test_social_pack_reports_spirit_stars_with_disclaimer() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:social", current_year=2026)

    joined = "\n".join(pack.facts)
    assert "天乙贵人在亥、酉，四柱未见" in joined
    assert "桃花（自年支巳起）在午，见于时支" in joined
    assert "天德贵人（子月起）在巳，见于年柱" in joined
    assert "传统神煞" in joined and "仅作参考" in joined


def test_health_pack_reports_element_distribution() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:health", current_year=2026)

    joined = "\n".join(pack.facts)
    assert "火 5 处" in joined
    assert "金 1 处" in joined


def test_luck_timing_pack_reports_cycles_and_annual_pillars() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:luck-timing", current_year=2026)

    joined = "\n".join(pack.facts)
    assert "起运" in joined
    assert "流年（2026—2035）" in joined
    assert "丙午" in joined


def test_unknown_topic_returns_none() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    assert build_topic_fact_pack(chart(), "topic:nonexistent") is None
