from datetime import datetime

from lunar_python import Solar

from .schemas import BirthInput, ChartFacts, PillarFacts

STEM_META = {
    "甲": ("木", "阳"),
    "乙": ("木", "阴"),
    "丙": ("火", "阳"),
    "丁": ("火", "阴"),
    "戊": ("土", "阳"),
    "己": ("土", "阴"),
    "庚": ("金", "阳"),
    "辛": ("金", "阴"),
    "壬": ("水", "阳"),
    "癸": ("水", "阴"),
}

BRANCH_ELEMENT = {
    "子": "水",
    "丑": "土",
    "寅": "木",
    "卯": "木",
    "辰": "土",
    "巳": "火",
    "午": "火",
    "未": "土",
    "申": "金",
    "酉": "金",
    "戌": "土",
    "亥": "水",
}


class ChartCalculator:
    """Produces chart facts from a deterministic calendrical library."""

    def calculate(self, birth: BirthInput) -> ChartFacts:
        solar = Solar.fromYmdHms(
            birth.date.year,
            birth.date.month,
            birth.date.day,
            birth.time.hour,
            birth.time.minute,
            birth.time.second,
        )
        eight_char = solar.getLunar().getEightChar()

        pillars = [
            self._pillar(eight_char, "year", "年柱"),
            self._pillar(eight_char, "month", "月柱"),
            self._pillar(eight_char, "day", "日柱"),
            self._pillar(eight_char, "time", "时柱"),
        ]
        day_master = eight_char.getDayGan()
        element, yin_yang = STEM_META[day_master]
        uncertainties: list[str] = []
        if birth.time.hour in {22, 23, 0, 1}:
            uncertainties.append("出生时间接近日界或子时边界，不同流派的换日规则可能改变日柱。")
        uncertainties.append("首版按输入地的民用时间排盘，尚未校正真太阳时。")

        return ChartFacts(
            calculated_at=datetime.now(),
            birth=birth,
            pillars=pillars,
            day_master=day_master,
            day_master_element=element,
            day_master_yin_yang=yin_yang,
            calculation_basis="lunar-python 节气历法；按民用出生时间计算，不由大模型推算",
            uncertainties=uncertainties,
        )

    def _pillar(self, eight_char: object, key: str, label: str) -> PillarFacts:
        prefix = key.capitalize()
        stem = getattr(eight_char, f"get{prefix}Gan")()
        branch = getattr(eight_char, f"get{prefix}Zhi")()
        stem_element, yin_yang = STEM_META[stem]
        return PillarFacts(
            key=key,
            label=label,
            stem=stem,
            branch=branch,
            stem_element=stem_element,
            stem_yin_yang=yin_yang,
            branch_element=BRANCH_ELEMENT[branch],
            hidden_stems=list(getattr(eight_char, f"get{prefix}HideGan")()),
            ten_god_stem=str(getattr(eight_char, f"get{prefix}ShiShenGan")()),
            ten_god_branches=list(getattr(eight_char, f"get{prefix}ShiShenZhi")()),
            nayin=str(getattr(eight_char, f"get{prefix}NaYin")()),
        )
