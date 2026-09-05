from datetime import datetime
from typing import Literal, assert_never

from lunar_python import EightChar, Solar

from .schemas import (
    BirthInput,
    ChartFacts,
    ExposedStemFact,
    LuckCycleFact,
    LuckFacts,
    MonthCommandFacts,
    PatternCandidateFact,
    PillarFacts,
    RootFact,
    StructuralRelationFact,
)

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

STEM_COMBINATIONS = [("甲", "己"), ("乙", "庚"), ("丙", "辛"), ("丁", "壬"), ("戊", "癸")]
BRANCH_COMBINATIONS = [
    ("子", "丑"),
    ("寅", "亥"),
    ("卯", "戌"),
    ("辰", "酉"),
    ("巳", "申"),
    ("午", "未"),
]
BRANCH_CLASHES = [
    ("子", "午"),
    ("丑", "未"),
    ("寅", "申"),
    ("卯", "酉"),
    ("辰", "戌"),
    ("巳", "亥"),
]
BRANCH_HARMS = [("子", "未"), ("丑", "午"), ("寅", "巳"), ("卯", "辰"), ("申", "亥"), ("酉", "戌")]
PUNISHMENT_GROUPS = [("寅", "巳", "申"), ("丑", "未", "戌"), ("子", "卯")]
SELF_PUNISHMENTS = {"辰", "午", "酉", "亥"}
THREE_HARMONY_GROUPS = [
    ("申", "子", "辰"),
    ("亥", "卯", "未"),
    ("寅", "午", "戌"),
    ("巳", "酉", "丑"),
]
SEASONAL_GROUPS = [("亥", "子", "丑"), ("寅", "卯", "辰"), ("巳", "午", "未"), ("申", "酉", "戌")]

GENERATES = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
CONTROLS = {"木": "土", "土": "水", "水": "火", "火": "金", "金": "木"}
PATTERN_NAMES = {
    "正官": "正官格候选",
    "七杀": "七杀格候选",
    "正财": "正财格候选",
    "偏财": "偏财格候选",
    "正印": "正印格候选",
    "偏印": "偏印格候选",
    "食神": "食神格候选",
    "伤官": "伤官格候选",
    "比肩": "建禄格候选",
    "劫财": "月劫格候选",
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
        month_command, exposed_stems, roots, relations, pattern_candidates = self.analyze_structure(
            pillars, day_master
        )
        luck = self._luck_facts(eight_char, birth)
        uncertainties: list[str] = []
        if birth.time.hour in {22, 23, 0, 1}:
            uncertainties.append("出生时间接近日界或子时边界，不同流派的换日规则可能改变日柱。")
        uncertainties.append(
            "已使用输入的出生时间完成四柱与大运计算；当前按民用时间计算，未另作真太阳时调整。"
        )

        return ChartFacts(
            calculated_at=datetime.now(),
            birth=birth,
            pillars=pillars,
            day_master=day_master,
            day_master_element=element,
            day_master_yin_yang=yin_yang,
            month_command=month_command,
            exposed_stems=exposed_stems,
            roots=roots,
            structural_relations=relations,
            pattern_candidates=pattern_candidates,
            luck=luck,
            calculation_basis=(
                "lunar-python 节气历法；"
                f"按 {birth.timezone} 当地民用时间计算，未校正真太阳时；"
                "大运按年干阴阳与性别定顺逆、按分钟法三天折一年；不由大模型推算"
            ),
            uncertainties=uncertainties,
        )

    @staticmethod
    def _luck_facts(eight_char: EightChar, birth: BirthInput) -> LuckFacts:
        yun = eight_char.getYun(1 if birth.gender == "male" else 0, sect=2)
        current_year = datetime.now().year
        cycles = [
            LuckCycleFact(
                index=item.getIndex(),
                ganzhi=item.getGanZhi(),
                stem=item.getGanZhi()[0],
                branch=item.getGanZhi()[1],
                start_year=item.getStartYear(),
                end_year=item.getEndYear(),
                start_age=item.getStartAge(),
                end_age=item.getEndAge(),
                status=(
                    "past"
                    if current_year > item.getEndYear()
                    else "future"
                    if current_year < item.getStartYear()
                    else "current"
                ),
            )
            for item in yun.getDaYun(11)[1:]
        ]
        current_cycle = next((item for item in cycles if item.status == "current"), None)
        next_cycle = next(
            (
                item
                for item in cycles
                if item.index == (current_cycle.index + 1 if current_cycle else 1)
            ),
            None,
        )
        return LuckFacts(
            direction="forward" if yun.isForward() else "reverse",
            direction_label="顺排" if yun.isForward() else "逆排",
            start_at=datetime.strptime(yun.getStartSolar().toYmdHms(), "%Y-%m-%d %H:%M:%S"),
            start_offset_years=yun.getStartYear(),
            start_offset_months=yun.getStartMonth(),
            start_offset_days=yun.getStartDay(),
            start_offset_hours=yun.getStartHour(),
            method="出生年干阴阳与性别定顺逆；按分钟法三天折一年",
            cycles=cycles,
            current_cycle=current_cycle,
            next_cycle=next_cycle,
        )

    def analyze_structure(
        self, pillars: list[PillarFacts], day_master: str
    ) -> tuple[
        MonthCommandFacts,
        list[ExposedStemFact],
        list[RootFact],
        list[StructuralRelationFact],
        list[PatternCandidateFact],
    ]:
        """Derive observable structure without asserting strength or successful transformation."""

        by_key = {pillar.key: pillar for pillar in pillars}
        month = by_key["month"]
        main_hidden_stem = month.hidden_stems[0]
        month_command = MonthCommandFacts(
            branch=month.branch,
            main_hidden_stem=main_hidden_stem,
            ten_god=self._ten_god(day_master, main_hidden_stem),
        )

        visible_positions: dict[str, list[str]] = {}
        hidden_positions: dict[str, list[str]] = {}
        for pillar in pillars:
            visible_positions.setdefault(pillar.stem, []).append(pillar.key)
            for hidden in pillar.hidden_stems:
                hidden_positions.setdefault(hidden, []).append(pillar.key)
        exposed_stems = [
            ExposedStemFact(
                hidden_stem=stem,
                hidden_positions=positions,
                visible_positions=visible_positions[stem],
                outside_day_master=any(key != "day" for key in visible_positions[stem]),
            )
            for stem, positions in hidden_positions.items()
            if stem in visible_positions
        ]

        roots: list[RootFact] = []
        for visible in pillars:
            visible_element = STEM_META[visible.stem][0]
            for branch_pillar in pillars:
                for hidden in branch_pillar.hidden_stems:
                    hidden_element = STEM_META[hidden][0]
                    if hidden == visible.stem or hidden_element == visible_element:
                        roots.append(
                            RootFact(
                                stem=visible.stem,
                                stem_position=visible.key,
                                branch=branch_pillar.branch,
                                branch_position=branch_pillar.key,
                                hidden_stem=hidden,
                                relation="same_stem" if hidden == visible.stem else "same_element",
                            )
                        )

        relations = self._relations(pillars)
        candidates: list[PatternCandidateFact] = []
        seen_candidate_stems: set[str] = set()
        for index, hidden in enumerate(month.hidden_stems):
            outside_day_positions = [
                key for key in visible_positions.get(hidden, []) if key != "day"
            ]
            is_main_qi = index == 0
            if not is_main_qi and not outside_day_positions:
                continue
            ten_god = self._ten_god(day_master, hidden)
            if ten_god not in PATTERN_NAMES or hidden in seen_candidate_stems:
                continue
            seen_candidate_stems.add(hidden)
            basis = "month_main_qi" if is_main_qi else "month_hidden_stem_exposed"
            source_label = "本气" if is_main_qi else "藏干"
            exposure_note = (
                f"并在{','.join(outside_day_positions)}干透出"
                if outside_day_positions
                else "未在日干之外透出"
            )
            candidates.append(
                PatternCandidateFact(
                    name=PATTERN_NAMES[ten_god],
                    ten_god=ten_god,
                    source_stem=hidden,
                    basis=basis,
                    exposed_positions=outside_day_positions,
                    note=(
                        f"由月支{month.branch}的{source_label}{hidden}提出，{exposure_note}；"
                        "仅为候选，未判断成格、破格或化气。"
                    ),
                )
            )
        return month_command, exposed_stems, roots, relations, candidates

    @staticmethod
    def _ten_god(day_master: str, target: str) -> str:
        day_element, day_polarity = STEM_META[day_master]
        target_element, target_polarity = STEM_META[target]
        same_polarity = day_polarity == target_polarity
        if day_element == target_element:
            return "比肩" if same_polarity else "劫财"
        if GENERATES[day_element] == target_element:
            return "食神" if same_polarity else "伤官"
        if GENERATES[target_element] == day_element:
            return "偏印" if same_polarity else "正印"
        if CONTROLS[day_element] == target_element:
            return "偏财" if same_polarity else "正财"
        return "七杀" if same_polarity else "正官"

    @classmethod
    def _relations(cls, pillars: list[PillarFacts]) -> list[StructuralRelationFact]:
        relations: list[StructuralRelationFact] = []
        stem_positions: dict[str, list[str]] = {}
        branch_positions: dict[str, list[str]] = {}
        for pillar in pillars:
            stem_positions.setdefault(pillar.stem, []).append(pillar.key)
            branch_positions.setdefault(pillar.branch, []).append(pillar.key)

        def add_pairs(kind: str, label_suffix: str, pairs: list[tuple[str, str]]) -> None:
            for left, right in pairs:
                if left in branch_positions and right in branch_positions:
                    relations.append(
                        StructuralRelationFact(
                            kind=kind,
                            label=f"{left}{right}{label_suffix}",
                            members=[left, right],
                            positions=[*branch_positions[left], *branch_positions[right]],
                            note="记录结构关系，不据此单独断吉凶或认定化气。",
                        )
                    )

        for left, right in STEM_COMBINATIONS:
            if left in stem_positions and right in stem_positions:
                relations.append(
                    StructuralRelationFact(
                        kind="stem_combination",
                        label=f"{left}{right}合",
                        members=[left, right],
                        positions=[*stem_positions[left], *stem_positions[right]],
                        note="天干相合已出现，但是否化气须另审月令、通根与制化。",
                    )
                )
        add_pairs("branch_combination", "合", BRANCH_COMBINATIONS)
        add_pairs("branch_clash", "冲", BRANCH_CLASHES)
        add_pairs("branch_harm", "害", BRANCH_HARMS)

        present = set(branch_positions)
        for group in PUNISHMENT_GROUPS:
            members = [branch for branch in group if branch in present]
            if len(members) >= 2:
                relations.append(
                    StructuralRelationFact(
                        kind="branch_punishment",
                        label=f"{''.join(members)}刑候选",
                        members=members,
                        positions=[key for branch in members for key in branch_positions[branch]],
                        completeness="full" if len(members) == len(group) else "partial",
                        note="刑的组合已出现；仅记录候选，不直接推断事件。",
                    )
                )
        for branch in SELF_PUNISHMENTS:
            if len(branch_positions.get(branch, [])) >= 2:
                relations.append(
                    StructuralRelationFact(
                        kind="branch_punishment",
                        label=f"{branch}{branch}自刑候选",
                        members=[branch, branch],
                        positions=branch_positions[branch],
                        note="自刑支重复出现；仅记录候选，不直接推断事件。",
                    )
                )

        for kind, groups, suffix in (
            ("three_harmony", THREE_HARMONY_GROUPS, "三合"),
            ("seasonal_meeting", SEASONAL_GROUPS, "三会"),
        ):
            for group in groups:
                members = [branch for branch in group if branch in present]
                if len(members) >= 2:
                    complete = len(members) == len(group)
                    relations.append(
                        StructuralRelationFact(
                            kind=kind,
                            label=f"{''.join(members)}{'全' if complete else '半'}{suffix}候选",
                            members=members,
                            positions=[
                                key for branch in members for key in branch_positions[branch]
                            ],
                            completeness="full" if complete else "partial",
                            note="组合条件已出现；是否成局或化气仍须审月令、透干和破坏条件。",
                        )
                    )
        return relations

    def _pillar(
        self,
        eight_char: EightChar,
        key: Literal["year", "month", "day", "time"],
        label: str,
    ) -> PillarFacts:
        if key == "year":
            stem = eight_char.getYearGan()
            branch = eight_char.getYearZhi()
            hidden_stems = eight_char.getYearHideGan()
            ten_god_stem = eight_char.getYearShiShenGan()
            ten_god_branches = eight_char.getYearShiShenZhi()
            nayin = eight_char.getYearNaYin()
        elif key == "month":
            stem = eight_char.getMonthGan()
            branch = eight_char.getMonthZhi()
            hidden_stems = eight_char.getMonthHideGan()
            ten_god_stem = eight_char.getMonthShiShenGan()
            ten_god_branches = eight_char.getMonthShiShenZhi()
            nayin = eight_char.getMonthNaYin()
        elif key == "day":
            stem = eight_char.getDayGan()
            branch = eight_char.getDayZhi()
            hidden_stems = eight_char.getDayHideGan()
            ten_god_stem = eight_char.getDayShiShenGan()
            ten_god_branches = eight_char.getDayShiShenZhi()
            nayin = eight_char.getDayNaYin()
        elif key == "time":
            stem = eight_char.getTimeGan()
            branch = eight_char.getTimeZhi()
            hidden_stems = eight_char.getTimeHideGan()
            ten_god_stem = eight_char.getTimeShiShenGan()
            ten_god_branches = eight_char.getTimeShiShenZhi()
            nayin = eight_char.getTimeNaYin()
        else:
            assert_never(key)
        stem_element, yin_yang = STEM_META[stem]
        return PillarFacts(
            key=key,
            label=label,
            stem=stem,
            branch=branch,
            stem_element=stem_element,
            stem_yin_yang=yin_yang,
            branch_element=BRANCH_ELEMENT[branch],
            hidden_stems=list(hidden_stems),
            ten_god_stem=str(ten_god_stem),
            ten_god_branches=list(ten_god_branches),
            nayin=str(nayin),
        )
    LuckCycleFact,
    LuckFacts,
