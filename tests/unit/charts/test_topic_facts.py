from datetime import date, time

from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.charts.topic_facts import (
    annual_ganzhi,
    annual_pillars,
    current_annual_year,
    element_distribution,
    lichun_date,
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
    # 每一行标注该流年的起算日：立春当天。
    assert [item["lichun"] for item in pillars] == [
        date(2026, 2, 4),
        date(2027, 2, 4),
        date(2028, 2, 4),
        date(2029, 2, 3),
        date(2030, 2, 4),
        date(2031, 2, 4),
        date(2032, 2, 4),
        date(2033, 2, 3),
        date(2034, 2, 4),
        date(2035, 2, 4),
    ]


def test_lichun_dates_follow_the_solar_terms_table() -> None:
    assert lichun_date(2025) == date(2025, 2, 3)
    assert lichun_date(2026) == date(2026, 2, 4)
    assert lichun_date(2033) == date(2033, 2, 3)


def test_current_annual_year_switches_at_lichun() -> None:
    assert current_annual_year(date(2026, 1, 20)) == 2025
    assert current_annual_year(date(2026, 2, 3)) == 2025
    assert current_annual_year(date(2026, 2, 4)) == 2026
    assert current_annual_year(date(2026, 6, 1)) == 2026
    assert current_annual_year(date(2026, 12, 31)) == 2026


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
    assert tian_yi_branches("壬") == ["卯", "巳"]
    assert tian_yi_branches("癸") == ["卯", "巳"]
    assert wenchang_branch("丙") == "申"
    assert peach_blossom_branches("巳") == ["午"]
    assert peach_blossom_branches("寅") == ["卯"]
    assert tian_de_target("子") == "巳"
    assert yue_de_target("子") == "壬"


def test_wealth_pack_reports_star_positions_and_counts() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:wealth", today=date(2026, 6, 1))

    assert pack is not None
    assert pack.topic_id == "topic:wealth"
    joined = "\n".join(pack.facts)
    assert "偏财（庚金）藏于年支巳" in joined
    assert "正财（辛金）四柱未见" in joined
    assert "仅为计数对比" in joined
    assert any("流年" in fact and "2026" in fact for fact in pack.facts)


def test_career_pack_reports_officer_and_seal_stars() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:career", today=date(2026, 6, 1))

    joined = "\n".join(pack.facts)
    assert "正官（癸水）藏于月支子" in joined
    assert "七杀（壬水）四柱未见" in joined
    assert "偏印（甲木）透于时柱" in joined


def test_relationship_pack_reports_spouse_palace_by_gender() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:relationships", today=date(2026, 6, 1))

    joined = "\n".join(pack.facts)
    assert "配偶宫为日支寅（藏干甲、丙、戊）" in joined
    assert "寅午半三合候选" in joined or "寅巳刑候选" in joined
    assert "男命看财星" in joined and "女命看官杀" in joined


def test_social_pack_reports_spirit_stars_with_disclaimer() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:social", today=date(2026, 6, 1))

    joined = "\n".join(pack.facts)
    assert "天乙贵人（日干丙起）在亥、酉，四柱未见" in joined
    assert "桃花（自年支巳起）在午，见于时支" in joined
    assert "天德贵人（子月起）在巳，见于年柱" in joined
    assert "传统神煞" in joined and "仅作参考" in joined


def test_family_pack_reports_seal_and_peer_stars_without_day_exposure() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:family", today=date(2026, 6, 1))

    assert pack is not None
    assert pack.topic_id == "topic:family"
    joined = "\n".join(pack.facts)
    # 印绶代长辈
    assert "偏印（甲木）透于时柱，又藏于日支寅" in joined
    assert "正印（乙木）四柱未见" in joined
    # 比劫代同辈：日柱天干即日主本人，不算“透出”
    assert "比肩（丙火）透于月柱，又藏于年支巳、日支寅" in joined
    assert "劫财（丁火）藏于时支午，未透干" in joined
    assert "透于月柱、日柱" not in joined
    assert not any("透" in fact and "日柱" in fact for fact in pack.facts)
    assert "十神配六亲为传统取象方法" in joined


def test_study_pack_reports_seal_stars_and_wenchang() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:study", today=date(2026, 6, 1))

    assert pack is not None
    assert pack.topic_id == "topic:study"
    joined = "\n".join(pack.facts)
    assert "偏印（甲木）透于时柱，又藏于日支寅" in joined
    assert "正印（乙木）四柱未见" in joined
    assert "文昌（日干丙起）在申，四柱未见。" in joined
    assert "传统神煞" in joined and "仅作参考" in joined
    assert any("流年" in fact and "2026" in fact for fact in pack.facts)


def test_health_pack_reports_element_distribution() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:health", today=date(2026, 6, 1))

    joined = "\n".join(pack.facts)
    assert "火 5 处" in joined
    assert "金 1 处" in joined


def test_luck_timing_pack_reports_cycles_and_annual_pillars() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:luck-timing", today=date(2026, 6, 1))

    joined = "\n".join(pack.facts)
    assert "起运" in joined
    assert "流年（2026 立春—2035 立春）" in joined
    assert "2026-02-04 立春起：丙午" in joined
    assert "2026-02-04 立春起：丙午、2027-02-04 立春起：丁未" in joined
    assert "流年按立春换年" in joined


def test_annual_line_before_lichun_still_belongs_to_previous_year() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:luck-timing", today=date(2026, 1, 20))

    assert pack is not None
    joined = "\n".join(pack.facts)
    # 2026-01-20 在立春之前，当前流年仍是乙巳（2025 立春年）。
    assert "流年（2025 立春—2034 立春）" in joined
    assert "2025-02-03 立春起：乙巳" in joined


def test_annual_line_after_lichun_uses_the_current_year() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    pack = build_topic_fact_pack(chart(), "topic:luck-timing", today=date(2026, 6, 1))

    assert pack is not None
    joined = "\n".join(pack.facts)
    # 2026-06-01 在立春之后，当前流年为丙午（2026 立春年）。
    assert "流年（2026 立春—2035 立春）" in joined
    assert "2026-02-04 立春起：丙午" in joined


def test_unknown_topic_returns_none() -> None:
    from bazi_api.modules.charts.topic_facts import build_topic_fact_pack

    assert build_topic_fact_pack(chart(), "topic:nonexistent") is None
