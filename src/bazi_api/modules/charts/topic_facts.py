"""话题相关的确定性事实：流年干支、五行分布与传统神煞查表。

只做查表与计数陈述；神煞按常见口诀录入，不同流派或有出入，输出时须带提示。
"""
from __future__ import annotations

from .schemas import PillarFacts
from .service import STEM_META

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
    "壬": ["巳", "卯"], "癸": ["巳", "卯"],
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
