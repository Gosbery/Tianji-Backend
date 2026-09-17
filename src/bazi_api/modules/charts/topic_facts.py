"""话题相关的确定性事实：流年干支、五行分布与传统神煞查表。

只做查表与计数陈述；神煞按常见口诀录入，不同流派或有出入，输出时须带提示。
"""
from __future__ import annotations

from datetime import datetime

from .schemas import ChartFacts, PillarFacts, TopicFactPack
from .service import CONTROLS, STEM_META, ChartCalculator

HEAVENLY_STEMS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
EARTHLY_BRANCHES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
ELEMENT_ORDER = ["木", "火", "土", "金", "水"]
# 1984 年为甲子年，作为六十甲子锚点。
JIAZI_ANCHOR_YEAR = 1984

# 天乙贵人（日干 → 地支，按“甲戊庚牛羊…”常用口诀）。
TIAN_YI_NOBLE: dict[str, list[str]] = {
    "甲": ["丑", "未"], "戊": ["丑", "未"], "庚": ["丑", "未"],
    "乙": ["子", "申"], "己": ["子", "申"],
    "丙": ["亥", "酉"], "丁": ["亥", "酉"],
    "壬": ["卯", "巳"], "癸": ["卯", "巳"],
    "辛": ["午", "寅"],
}
# 文昌（日干 → 地支）。
WENCHANG: dict[str, str] = {
    "甲": "巳", "乙": "午", "丙": "申", "戊": "申", "丁": "酉", "己": "酉",
    "庚": "亥", "辛": "子", "壬": "寅", "癸": "卯",
}
# 桃花：年支或日支所在三合局 → 桃花支。
PEACH_BLOSSOM_GROUPS: dict[tuple[str, str, str], str] = {
    ("申", "子", "辰"): "酉",
    ("寅", "午", "戌"): "卯",
    ("巳", "酉", "丑"): "午",
    ("亥", "卯", "未"): "子",
}
# 月德：月支三合局 → 天干。天德：月支 → 干支（常见口诀，流派或有出入）。
YUE_DE_GROUPS: dict[tuple[str, str, str], str] = {
    ("寅", "午", "戌"): "丙",
    ("申", "子", "辰"): "壬",
    ("亥", "卯", "未"): "甲",
    ("巳", "酉", "丑"): "庚",
}
TIAN_DE: dict[str, str] = {
    "寅": "丁", "卯": "申", "辰": "壬", "巳": "辛", "午": "亥", "未": "甲",
    "申": "癸", "酉": "寅", "戌": "丙", "亥": "乙", "子": "巳", "丑": "庚",
}


def annual_ganzhi(year: int) -> str:
    offset = (year - JIAZI_ANCHOR_YEAR) % 60
    return HEAVENLY_STEMS[offset % 10] + EARTHLY_BRANCHES[offset % 12]


def annual_pillars(start_year: int, years: int = 10) -> list[dict[str, int | str]]:
    return [
        {"year": year, "ganzhi": annual_ganzhi(year)}
        for year in range(start_year, start_year + years)
    ]


def element_distribution(pillars: list[PillarFacts]) -> list[str]:
    counts: dict[str, int] = {element: 0 for element in ELEMENT_ORDER}
    for pillar in pillars:
        counts[STEM_META[pillar.stem][0]] += 1
        for hidden in pillar.hidden_stems:
            counts[STEM_META[hidden][0]] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], ELEMENT_ORDER.index(item[0])))
    return [f"{element} {count} 处" for element, count in ordered]


def tian_yi_branches(day_stem: str) -> list[str]:
    return list(TIAN_YI_NOBLE[day_stem])


def wenchang_branch(day_stem: str) -> str:
    return WENCHANG[day_stem]


def peach_blossom_branches(branch: str) -> list[str]:
    return [target for group, target in PEACH_BLOSSOM_GROUPS.items() if branch in group]


def tian_de_target(month_branch: str) -> str:
    return TIAN_DE[month_branch]


def yue_de_target(month_branch: str) -> str:
    for group, target in YUE_DE_GROUPS.items():
        if month_branch in group:
            return target
    raise KeyError(month_branch)


TOPIC_LABELS: dict[str, str] = {
    "topic:wealth": "财富与财运",
    "topic:career": "事业与职业",
    "topic:relationships": "感情与婚恋",
    "topic:health": "健康文献观点",
    "topic:social": "人际与合作",
    "topic:study": "学业与考试",
    "topic:luck-timing": "大运流年与趋势",
    "topic:family": "六亲与家庭",
}
SPIRIT_STAR_NOTE = "以上为传统神煞查表口诀，不同流派或有出入，仅作参考。"


def _ten_god_of(day_master: str, target: str) -> str:
    return ChartCalculator._ten_god(day_master, target)


def _god_stems(day_master: str, gods: tuple[str, ...]) -> dict[str, str]:
    """返回 {天干: 十神}，仅包含命中给定十神集合的天干。"""
    return {
        stem: _ten_god_of(day_master, stem)
        for stem in HEAVENLY_STEMS
        if _ten_god_of(day_master, stem) in gods
    }


def _star_report(chart: ChartFacts, stems: dict[str, str]) -> list[str]:
    """按天干逐一陈述配星的透干与藏支位置，只报结构，不判定吉凶。

    “透干”指相对日主而言在四柱天干出现；日柱天干恒等于日主本人，不算透出
    （与 ``charts/service.py`` 的 ``outside_day_master`` 口径一致）。
    """
    lines: list[str] = []
    for stem, god in stems.items():
        visible = [
            pillar.label
            for pillar in chart.pillars
            if pillar.stem == stem and pillar.key != "day"
        ]
        hidden = [
            f"{pillar.label.replace('柱', '支')}{pillar.branch}"
            for pillar in chart.pillars
            if stem in pillar.hidden_stems
        ]
        meta = STEM_META[stem][0]
        if visible and hidden:
            lines.append(
                f"{god}（{stem}{meta}）透于{'、'.join(visible)}，又藏于{'、'.join(hidden)}"
            )
        elif visible:
            lines.append(f"{god}（{stem}{meta}）透于{'、'.join(visible)}")
        elif hidden:
            lines.append(f"{god}（{stem}{meta}）藏于{'、'.join(hidden)}，未透干")
        else:
            lines.append(f"{god}（{stem}{meta}）四柱未见")
    return lines


def _same_element_count(chart: ChartFacts, element: str) -> int:
    count = sum(1 for pillar in chart.pillars if STEM_META[pillar.stem][0] == element)
    count += sum(
        1
        for pillar in chart.pillars
        for hidden in pillar.hidden_stems
        if STEM_META[hidden][0] == element
    )
    return count


def _found_positions(chart: ChartFacts, stem_or_branch: str) -> str:
    """返回命中该天干或地支的柱标签（年柱/月柱/日柱/时柱），未命中返回空串。"""
    hits = [
        pillar.label
        for pillar in chart.pillars
        if pillar.stem == stem_or_branch or pillar.branch == stem_or_branch
    ]
    return "、".join(hits)


def _annual_line(current_year: int) -> str:
    pillars = annual_pillars(current_year, 10)
    table = "、".join(f"{item['year']}{item['ganzhi']}" for item in pillars)
    return (
        f"流年（{current_year}—{current_year + 9}）干支依次为：{table}。"
        "流年由六十甲子推算，属程序计算事实。"
    )


def _spirit_star_lines(chart: ChartFacts) -> list[str]:
    """按各条口诀起例并给出命中位置：贵人与桃花报地支，天德月德报柱。"""
    lines: list[str] = []
    branches_by_key = {pillar.key: pillar.branch for pillar in chart.pillars}
    all_branches = [pillar.branch for pillar in chart.pillars]

    noble = tian_yi_branches(chart.day_master)
    present = [
        f"{'年月日时'[index]}支" for index, branch in enumerate(all_branches) if branch in noble
    ]
    label = "、".join(noble)
    if present:
        lines.append(f"天乙贵人（日干{chart.day_master}起）在{label}，见于{'、'.join(present)}。")
    else:
        lines.append(f"天乙贵人（日干{chart.day_master}起）在{label}，四柱未见。")

    year_branch = branches_by_key["year"]
    day_branch = branches_by_key["day"]
    year_peach = peach_blossom_branches(year_branch)
    day_peach = peach_blossom_branches(day_branch)
    peach_hits: list[str] = []
    for index, branch in enumerate(all_branches):
        if branch in year_peach or branch in day_peach:
            source = f"年支{year_branch}" if branch in year_peach else f"日支{day_branch}"
            peach_hits.append(f"桃花（自{source}起）在{branch}，见于{'年月日时'[index]}支。")
    lines.append("".join(peach_hits) if peach_hits else "桃花星四柱未见。")

    month_branch = branches_by_key["month"]
    tian_de = tian_de_target(month_branch)
    tian_de_hits = _found_positions(chart, tian_de)
    lines.append(
        f"天德贵人（{month_branch}月起）在{tian_de}，见于{tian_de_hits}。"
        if tian_de_hits
        else f"天德贵人（{month_branch}月起）在{tian_de}，四柱未见。"
    )
    yue_de = yue_de_target(month_branch)
    yue_de_hits = _found_positions(chart, yue_de)
    lines.append(
        f"月德贵人（{month_branch}月起）在{yue_de}，见于{yue_de_hits}。"
        if yue_de_hits
        else f"月德贵人（{month_branch}月起）在{yue_de}，四柱未见。"
    )
    lines.append(SPIRIT_STAR_NOTE)
    return lines


def _spouse_palace_lines(chart: ChartFacts) -> list[str]:
    day = next(pillar for pillar in chart.pillars if pillar.key == "day")
    hidden = "、".join(day.hidden_stems)
    relations = [
        relation.label for relation in chart.structural_relations if "day" in relation.positions
    ]
    return [
        f"配偶宫为日支{day.branch}（藏干{hidden}）。",
        f"配偶宫参与的结构关系：{'、'.join(relations) or '无'}；仅记录结构出现，不代表吉凶。",
        "男命看财星、女命看官杀为传统配星方法：",
    ]


def _luck_lines(chart: ChartFacts) -> list[str]:
    luck = chart.luck
    lines = [f"按{luck.direction_label}计算，起运时间{luck.start_at:%Y-%m-%d %H:%M}。"]
    for prefix, cycle in (("当前大运", luck.current_cycle), ("下一大运", luck.next_cycle)):
        if cycle is not None:
            lines.append(
                f"{prefix}{cycle.ganzhi}（{cycle.start_year}—{cycle.end_year} 年，"
                f"{cycle.start_age}—{cycle.end_age} 岁）。"
            )
    return lines


def build_topic_fact_pack(
    chart: ChartFacts, topic_id: str, *, current_year: int | None = None
) -> TopicFactPack | None:
    """按话题汇总确定性事实（查表与计数陈述），未知话题返回 None。"""
    label = TOPIC_LABELS.get(topic_id)
    if label is None:
        return None
    year = current_year or datetime.now().year
    facts: list[str] = []

    if topic_id == "topic:wealth":
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正财", "偏财"))))
        wealth_element = CONTROLS[chart.day_master_element]
        facts.append(
            f"与日主同五行（{chart.day_master_element}）干支 "
            f"{_same_element_count(chart, chart.day_master_element)} 处，"
            f"财星五行（{wealth_element}）干支 {_same_element_count(chart, wealth_element)} 处；"
            "仅为计数对比，不判定身强身弱。"
        )
        facts.append(_annual_line(year))
    elif topic_id == "topic:career":
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正官", "七杀"))))
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正印", "偏印"))))
        facts.append(_annual_line(year))
    elif topic_id == "topic:relationships":
        facts.extend(_spouse_palace_lines(chart))
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正财", "偏财"))))
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正官", "七杀"))))
        facts.append(_annual_line(year))
    elif topic_id == "topic:health":
        facts.append(
            "四柱五行分布（天干加藏干计数）："
            + "、".join(element_distribution(chart.pillars))
            + "。"
        )
        facts.append("五行分布仅为计数，健康相关结论只描述传统文献观点。")
    elif topic_id == "topic:social":
        facts.extend(_spirit_star_lines(chart))
        facts.append(_annual_line(year))
    elif topic_id == "topic:study":
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正印", "偏印"))))
        wenchang = wenchang_branch(chart.day_master)
        hits = _found_positions(chart, wenchang)
        facts.append(f"文昌（日干{chart.day_master}起）在{wenchang}，{hits or '四柱未见'}。")
        facts.append(SPIRIT_STAR_NOTE)
        facts.append(_annual_line(year))
    elif topic_id == "topic:luck-timing":
        facts.extend(_luck_lines(chart))
        facts.append(_annual_line(year))
    elif topic_id == "topic:family":
        # 印绶代长辈，比劫代同辈，均为传统取象。
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("正印", "偏印"))))
        facts.extend(_star_report(chart, _god_stems(chart.day_master, ("比肩", "劫财"))))
        facts.append("十神配六亲为传统取象方法，仅列结构事实。")

    return TopicFactPack(topic_id=topic_id, label=label, facts=facts)
