from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "knowledge" / "originals" / "ziping-zhenquan.yml"
OUTPUT = ROOT / "knowledge" / "cards" / "ziping-zhenquan-v2.yml"
PRIORITY_CHAPTERS = {*range(8, 21), *range(31, 48)}


CHAPTER_FOCUS = {
    1: "十干十二支的阴阳五行基础",
    2: "阴阳生克不可脱离强弱与配合",
    3: "十干生旺死绝须结合阴阳性质",
    4: "天干五合的性情与作用",
    5: "天干有合而不以合论的条件",
    6: "得时失时不等于绝对旺衰",
    7: "刑冲会合须按结构解读",
    8: "以月令为起点辨顺用逆用",
    9: "逐层判断成格、败格、带忌与救应",
    10: "月令藏干透会引起的用神变化",
    11: "兼用结构的纯杂与协同",
    12: "以有情、有力衡量格局层次",
    13: "成败会因新增关系再次反转",
    14: "月令格局必须与寒暖燥湿互参",
    15: "识别全局成格所赖的相神",
    16: "杂气月以透干会支取清",
    17: "墓库刑冲不等于冲开即发",
    18: "吉神使用失宜也会破格",
    19: "凶神制化得宜也能成格",
    20: "同样生克因位置先后而结果不同",
    21: "星辰神煞不得替代格局判断",
    22: "仅在月令无可用结构时讨论外格",
    23: "宫位与十神共同讨论六亲",
    24: "妻子判断须回到财官与全局",
    25: "行运须放回原局喜忌",
    26: "行运可能促成、破坏或改变格局",
    27: "喜忌在天干地支的作用有别",
    28: "支中喜忌可在行运透出而显效",
    29: "格局名称不可拘泥套用",
    30: "辨析缺乏结构依据的流传说法",
    31: "正官格的成格配合与禁忌",
    32: "正官格按原局配置选择行运",
    33: "财格的生官、食生、佩印与制劫",
    34: "财格按成格路径选择行运",
    35: "印绶格的官印、煞印、食泄与财损",
    36: "印绶格按成格路径选择行运",
    37: "食神格的生财、制煞与弃食就煞",
    38: "食神格按成格路径选择行运",
    39: "七杀格的食制、印化、财清与取纯",
    40: "七杀格按制化轻重选择行运",
    41: "伤官格结合财印官煞及气候取用",
    42: "伤官格按成格路径选择行运",
    43: "阳刃格以官杀制刃并审财印食伤",
    44: "阳刃格按官杀轻重选择行运",
    45: "建禄月劫另取财官杀食为用",
    46: "建禄月劫按成格路径选择行运",
    47: "外格必须满足严格资格且先排除正格",
}

LUCK_CHAPTERS = {25, 26, 27, 28, 32, 34, 36, 38, 40, 42, 44, 46}
PATTERN_CHAPTERS = {*range(31, 48)}
CASE_PASSAGES = {
    "ziping-zhenquan-ch31-p002",
    "ziping-zhenquan-ch31-p003",
    "ziping-zhenquan-ch33-p002",
    "ziping-zhenquan-ch33-p005",
    "ziping-zhenquan-ch35-p001",
    "ziping-zhenquan-ch35-p008",
    "ziping-zhenquan-ch37-p002",
    "ziping-zhenquan-ch39-p002",
    "ziping-zhenquan-ch41-p002",
    "ziping-zhenquan-ch43-p003",
    "ziping-zhenquan-ch45-p004",
    "ziping-zhenquan-ch47-p007",
}
PILLARS_RE = re.compile(
    r"([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
    r"[、，,]\s*([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
    r"[、，,]\s*([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
    r"[、，,]\s*([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥])"
)


def chapter_number(passage_id: str) -> int:
    match = re.search(r"-ch(\d+)-", passage_id)
    if not match:
        raise ValueError(f"missing chapter number: {passage_id}")
    return int(match.group(1))


def concise_claim(text: str) -> str | None:
    sentences = [item.strip() for item in re.split(r"(?<=[。！？])", text) if item.strip()]
    declarative = [item for item in sentences if not item.rstrip().endswith(("？", "?"))]
    if not declarative:
        return None
    claim = "".join(declarative[:2])
    if len(claim) <= 220:
        return claim
    return claim[:217].rstrip("，；：") + "……"


def source_reasoning_signals(text: str) -> dict[str, list[str]]:
    clauses = [
        clause.strip(" ，。；：")
        for clause in re.split(r"[。；]", text)
        if len(clause.strip(" ，。；：")) >= 6
    ]

    def select(markers: tuple[str, ...], limit: int = 2) -> list[str]:
        matches = [clause for clause in clauses if any(marker in clause for marker in markers)]
        return [f"原文条件：{clause}" for clause in matches[:limit]]

    return {
        "conditions": select(("须", "要", "喜", "宜", "若", "逢", "透", "会", "用")),
        "exceptions": select(("然", "但", "反", "亦有", "未必", "不可执"), 1),
        "break_conditions": select(
            (
                "切忌",
                "最忌",
                "忌见",
                "忌逢",
                "不喜",
                "不利",
                "畏",
                "怕",
                "破格",
                "格败",
                "不能",
                "不得",
                "无用",
                "无辅",
                "无制",
                "不成",
            )
        ),
        "rescue_conditions": select(
            ("救应", "以解", "护", "制伏", "合去", "存", "取清", "生财", "生官", "佩印")
        ),
        "counterexamples": select(("今人", "不知", "不可", "不作", "不以", "非"), 1),
    }


def merge_signals(defaults: dict[str, object], signals: dict[str, list[str]]) -> dict[str, object]:
    merged = dict(defaults)
    for field, source_values in signals.items():
        fallback = list(merged[field])
        merged[field] = [*source_values, *fallback][:3]
    return merged


def reasoning_fields(chapter: int) -> dict[str, object]:
    focus = CHAPTER_FOCUS[chapter]
    if chapter in LUCK_CHAPTERS:
        return {
            "premises": [
                "原局格局、用神、相神与忌神已有可追溯判断",
                "当前只讨论行运如何作用于原局",
            ],
            "conditions": [
                f"逐项核对运干运支如何影响{focus}",
                "区分助成格之神、损成格之神与取清之神",
            ],
            "exceptions": ["同一五行在不同原局或不同成格路径中喜忌可以相反"],
            "break_conditions": [
                "脱离原局只凭运支或运干固定断吉凶",
                "忽略原局已有的合冲和强弱条件",
            ],
            "rescue_conditions": ["先恢复原局成格路径，再判断此运补足、制伏、通关或取清何者"],
            "counterexamples": ["见到财运便一律断财吉，未检查财在该格中是否党煞、坏印或泄食"],
            "priority": 72,
            "priority_note": "行运规则必须后于原局结构判断，不能反过来用流年运字定义原局。",
        }
    if chapter == 47:
        return {
            "premises": [
                "已先检查月令、透干和会支，未发现可成立的常规格局",
                "逐项满足该外格原文列明的资格",
            ],
            "conditions": [
                "核对四柱是否齐备所需字形、月令与无官杀等限制",
                "先排除财官印食杀伤等更直接取用",
            ],
            "exceptions": ["原文明确废弃或存疑的名目不得作为可靠外格"],
            "break_conditions": ["官杀或根深之财已经可直接为用", "只满足外格名称的一部分字形"],
            "rescue_conditions": ["退回月令正格重新取用", "保留为低置信候选并明确缺失条件"],
            "counterexamples": ["见到拱夹、遥合或四同字便强立外格，忽略已有官杀财星"],
            "priority": 55,
            "priority_note": "外格优先级低于月令正格，资格不全时必须放弃而非迁就。",
        }
    if chapter in PATTERN_CHAPTERS or 8 <= chapter <= 20:
        return {
            "premises": ["以日主配月令提出格局候选", "四柱透干、会支、位置与根气已经由程序列明"],
            "conditions": [
                f"围绕{focus}逐项核对原文列出的配合",
                "同时检查成格要素、忌神、相神和力量是否到位",
            ],
            "exceptions": ["同名十神在不同月令格局中作用不同，不按名称固定判吉凶"],
            "break_conditions": [
                "关键成格要素被合去、冲伤或受制",
                "只见一个十神便跳过月令和全局配合直接定格",
            ],
            "rescue_conditions": [
                "检查原局是否另有制忌、合忌、通关、护用或取清关系",
                "有救应时重新评估，而非沿用初步败格结论",
            ],
            "counterexamples": [f"只因命局出现相关十神便宣称符合“{focus}”，却未满足本条所列条件"],
            "priority": 88 if 8 <= chapter <= 20 else 82,
            "priority_note": "本条属于格局判断主链；先确认月令与结构事实，再依成败救应顺序应用。",
        }
    return {
        "premises": ["四柱干支、月令和日主已经准确排定", "当前采用《子平真诠》的月令格局语境"],
        "conditions": [f"仅在讨论{focus}时应用本条", "把本条与后续成败、救应和气候规则共同核对"],
        "exceptions": ["原文未覆盖的流派定义或现代延伸不得自动并入"],
        "break_conditions": ["脱离月令与四柱位置孤立套用术语"],
        "rescue_conditions": ["退回可验证的干支事实", "补齐月令、透藏、合冲与上下文后再判断"],
        "counterexamples": [f"只见关键词便断定已满足“{focus}”，未核对本条的上下文"],
        "priority": 65,
        "priority_note": "本条提供基础或边界规则，须服从月令、成败救应与全局配合。",
    }


def make_rule(passage: dict[str, object]) -> dict[str, object] | None:
    passage_id = str(passage["id"])
    chapter = chapter_number(passage_id)
    text = str(passage["text"])
    paragraph = int(re.search(r"-p(\d+)$", passage_id).group(1))  # type: ignore[union-attr]
    focus = CHAPTER_FOCUS[chapter]
    conclusion = concise_claim(text)
    question_only = conclusion is None
    if question_only:
        question = text.rstrip("？?").strip()
        conclusion = f"本段仅提出“{question}”这一待辨问题，须结合本章后续段落后形成规则结论。"
    fields = merge_signals(reasoning_fields(chapter), source_reasoning_signals(text))
    return {
        "id": f"ziping-rule-ch{chapter:02d}-p{paragraph:03d}",
        "title": f"第{chapter}章规则{paragraph}：{focus}",
        "schema_version": 2,
        "card_type": (
            "boundary" if question_only else ("method" if chapter in LUCK_CHAPTERS else "rule")
        ),
        "content": text,
        "rule": f"在“{focus}”的判断链中，应核验本段结论：{conclusion}",
        "school": "子平格局法",
        "concepts": list(passage.get("concepts", [])) or ["格局"],
        "conclusion": conclusion,
        **fields,
        "exclusions": [],
        "prohibited_uses": ["不得省略前提与破格条件只引用结论", "不得把机器校勘内容标作人工审核"],
        "source_refs": [{"passage_id": passage_id}],
        "graph_refs": list(passage.get("graph_refs", [])),
        "status": "machine_verified",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_pages": list(passage.get("source_pages", [])),
        "collation_method": "source_passage_binding+explicit_reasoning_fields+schema_v2_validation",
        "source_count": int(passage.get("source_count", 1)),
        "verification_level": str(passage.get("verification_level", "single_source_integrity")),
        "confidence": min(float(passage.get("confidence", 0.56)), 0.62),
        "unresolved_variants": list(passage.get("unresolved_variants", [])),
    }


def make_case(passage: dict[str, object]) -> dict[str, object]:
    passage_id = str(passage["id"])
    chapter = chapter_number(passage_id)
    text = str(passage["text"])
    match = PILLARS_RE.search(text)
    if not match:
        raise ValueError(f"selected case has no valid four pillars: {passage_id}")
    pillars = list(match.groups())
    paragraph = int(re.search(r"-p(\d+)$", passage_id).group(1))  # type: ignore[union-attr]
    content = f"原典命例：{' / '.join(pillars)}。{text}"
    return {
        "id": f"ziping-case-ch{chapter:02d}-p{paragraph:03d}",
        "title": f"典型命例：第{chapter}章{CHAPTER_FOCUS[chapter]}",
        "schema_version": 2,
        "card_type": "case",
        "content": content,
        "school": "子平格局法",
        "concepts": list(passage.get("concepts", [])) or ["命例", "格局"],
        "conclusion": (
            f"此例在原书中用于说明{CHAPTER_FOCUS[chapter]}；"
            "只可作为结构应用样本，不把古代身份或富贵评价机械移植到其他命局。"
        ),
        "rule_refs": [f"ziping-rule-ch{chapter:02d}-p{paragraph:03d}"],
        "case_pillars": pillars,
        "application_steps": [
            "先由确定性计算确认日主、月令、透干、根气及合冲刑害",
            f"定位第{chapter}章所讨论的结构候选，不先套用命例结论",
            "对照所引规则逐项检查前提、成格要素、破格因素与救应",
            "只保留与当前四柱事实相同的推理步骤，并明确不同之处",
        ],
        "exclusions": [],
        "prohibited_uses": ["不得只因一柱或一字相同就类推整个人生"],
        "source_refs": [{"passage_id": passage_id}],
        "graph_refs": list(passage.get("graph_refs", [])),
        "status": "machine_verified",
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_pages": list(passage.get("source_pages", [])),
        "collation_method": "canonical_case_extraction+pillar_syntax_validation+rule_linkage",
        "source_count": int(passage.get("source_count", 1)),
        "verification_level": str(passage.get("verification_level", "single_source_integrity")),
        "confidence": min(float(passage.get("confidence", 0.56)), 0.6),
        "unresolved_variants": list(passage.get("unresolved_variants", [])),
    }


def main() -> None:
    corpus = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    passages = [
        passage
        for passage in corpus["passages"]
        if passage.get("volume") == "正文" and "-ch" in passage["id"]
    ]
    selected: list[dict[str, object]] = []
    seen_nonpriority: set[int] = set()
    for passage in passages:
        chapter = chapter_number(str(passage["id"]))
        if chapter in PRIORITY_CHAPTERS or chapter not in seen_nonpriority:
            selected.append(passage)
            seen_nonpriority.add(chapter)

    built_rules = [make_rule(passage) for passage in selected]
    cards = [card for card in built_rules if card is not None]
    rejected = [
        str(passage["id"])
        for passage, card in zip(selected, built_rules, strict=True)
        if card is None
    ]
    by_id = {str(passage["id"]): passage for passage in passages}
    cards.extend(make_case(by_id[passage_id]) for passage_id in sorted(CASE_PASSAGES))
    OUTPUT.write_text(
        yaml.safe_dump({"cards": cards}, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    unique_rules = len({str(card.get("rule", "")) for card in cards if card["card_type"] != "case"})
    print(
        f"wrote {len(cards)} cards ({len(cards) - len(CASE_PASSAGES)} rules, "
        f"{len(CASE_PASSAGES)} cases, {len(rejected)} question-only passages rejected, "
        f"{unique_rules} unique rule claims)"
    )


if __name__ == "__main__":
    main()
