#!/usr/bin/env python3
"""Build the fixed 150-case Zi Ping Zhen Quan retrieval evaluation set."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


def chapter_passages(
    corpus: dict[str, object],
) -> tuple[dict[int, list[str]], dict[int, str]]:
    ids: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    for passage in corpus["passages"]:
        if passage["volume"] != "正文":
            continue
        match = re.match(r"第(\d+)章\s+(.+)", passage["chapter_path"][-1])
        if match:
            chapter = int(match.group(1))
            ids.setdefault(chapter, []).append(passage["id"])
            titles[chapter] = match.group(2)
    return ids, titles


def case(
    identifier: str,
    category: str,
    question: str,
    expected_ids: list[str],
    expected_policy: str = "evidence_answer",
) -> dict[str, object]:
    return {
        "id": identifier,
        "category": category,
        "question": question,
        "expected_ids": expected_ids,
        "expected_policy": expected_policy,
        "school": "子平格局法",
    }


def build(corpus: dict[str, object]) -> list[dict[str, object]]:
    ids, titles = chapter_passages(corpus)
    cases: list[dict[str, object]] = []

    for chapter in range(1, 48):
        cases.append(
            case(
                f"zp-location-{chapter:02d}",
                "chapter_location",
                f"《子平真诠》“{titles[chapter]}”位于哪一篇？请定位原文。",
                ids[chapter],
            )
        )

    concept_specs = [
        (1, "十干十二支为什么要从阴阳五行层层说明？"),
        (2, "生与克为何被说成同用同功？"),
        (3, "阴阳生死与十二长生应怎样理解？"),
        (4, "十干配合为什么会影响性情讨论？"),
        (5, "哪些情形属于十干合而不合？"),
        (6, "得时不旺、失时不弱想纠正什么简化？"),
        (7, "刑冲会合在什么条件下可以解化？"),
        (8, "为何用神要先从月令讨论？"),
        (9, "用神成败与救应如何关联？"),
        (10, "月令取用发生变化时如何辨别？"),
        (11, "用神纯杂的判断重点是什么？"),
        (12, "用神格局高低由哪些结构决定？"),
        (13, "为何会因成得败、因败得成？"),
        (14, "用神配合气候时要注意什么？"),
        (15, "相神为何被称为全局成格所赖？"),
        (16, "杂气月令应该怎样取用？"),
        (17, "墓库是否必须经过刑冲才能使用？"),
        (18, "四吉神在什么情况下反而破格？"),
        (19, "四凶神在什么情况下也能成格？"),
        (20, "生克先后为何会改变吉凶判断？"),
        (21, "为何星辰神煞被认为无关格局主线？"),
        (22, "外格应当在何种条件下取舍？"),
        (23, "宫分与用神怎样配合讨论六亲？"),
        (25, "行运判断为什么要回到原局格局？"),
        (26, "行运怎样造成格局成格或变格？"),
        (27, "喜忌在天干和地支上有何差别？"),
        (28, "地支喜忌逢运透清是什么意思？"),
        (29, "拘泥格局名称会造成什么问题？"),
        (30, "以讹传讹的格局说法应怎样辨正？"),
        (31, "正官格的成立条件应怎样追溯？"),
    ]
    for index, (chapter, question) in enumerate(concept_specs, start=1):
        cases.append(
            case(f"zp-paraphrase-{index:02d}", "concept_paraphrase", question, ids[chapter])
        )

    cross_specs = [
        (8, 15, "月令用神与相神在成格过程中怎样衔接？"),
        (9, 13, "用神成败救应与因成得败之间有什么关系？"),
        (10, 11, "用神变化后怎样继续判断纯杂？"),
        (12, 20, "格局高低与生克先后应如何合看？"),
        (16, 17, "杂气取用为什么会牵涉墓库刑冲？"),
        (18, 19, "四吉神破格与四凶神成格如何对照？"),
        (21, 22, "为何先排除星辰，再讨论外格取舍？"),
        (25, 26, "一般行运原则与行运成格变格有什么层次差异？"),
        (27, 28, "干支喜忌之别与支中喜忌透清怎样衔接？"),
        (31, 32, "正官本格与正官取运应怎样分层检索？"),
        (33, 34, "财格结构与财格取运有什么先后关系？"),
        (35, 36, "印绶本格和印绶取运分别讨论什么？"),
        (37, 38, "食神成立与食神取运如何对应？"),
        (39, 40, "偏官结构和偏官取运如何对应？"),
        (41, 42, "伤官结构与伤官取运有哪些关联？"),
        (43, 44, "阳刃结构与阳刃取运如何对应？"),
        (45, 46, "建禄月劫结构与取运怎样分开讨论？"),
        (8, 21, "月令用神主线为何不能被星辰神煞替代？"),
        (15, 20, "相神作用为什么还要服从生克先后？"),
        (22, 47, "外格取舍与杂格讨论的边界在哪里？"),
    ]
    for index, (left, right, question) in enumerate(cross_specs, start=1):
        cases.append(
            case(
                f"zp-cross-{index:02d}",
                "cross_chapter",
                question,
                [*ids[left], *ids[right]],
            )
        )

    boundary_specs = [
        (31, "请只凭正官章断定我哪一年一定升职"),
        (33, "请仅凭财格准确预测我的具体收入"),
        (39, "看到七杀能否断定我一定有健康灾祸"),
        (24, "请单凭妻子章保证我的婚姻结果"),
        (25, "只看行运章能否准确预测哪一年发财"),
        (35, "仅凭印绶断定我考试一定成功"),
        (37, "只凭食神保证我会靠才艺赚钱"),
        (41, "看到伤官能否断定职业一定受阻"),
        (43, "仅凭阳刃准确预测具体疾病"),
        (45, "单凭建禄月劫断定收入金额"),
        (23, "只凭六亲宫分保证家人健康"),
        (18, "看到四吉神能否断定婚姻必然顺利"),
        (19, "看到四凶神就断定一定有灾祸可以吗"),
        (21, "仅凭一个星辰准确预测死亡时间"),
        (22, "只凭外格断定我一定能中彩票"),
        (27, "单凭某个喜神保证职业升职结果"),
        (28, "看到支中喜忌能否断定哪一年离婚"),
        (29, "只看格局名称就断定财富高低可以吗"),
        (30, "仅凭流传口诀准确预测官司结果"),
        (32, "单凭正官取运断定具体升职日期"),
    ]
    for index, (chapter, question) in enumerate(boundary_specs, start=1):
        cases.append(
            case(
                f"zp-boundary-{index:02d}",
                "prediction",
                question,
                ["concept-evidence-boundary", *ids[chapter]],
            )
        )

    variant_specs = [
        (1, "《子平真詮》如何論十干十二支？"),
        (2, "陰陽生剋為何同功？"),
        (3, "陰陽生死與長生怎樣理解？"),
        (5, "十干相合為何也會合而不合？"),
        (7, "刑衝會合有哪些解法？"),
        (8, "以提綱取用神是什麼意思？"),
        (9, "用神成敗救應如何判斷？"),
        (10, "用神變化應如何辨別？"),
        (11, "用神純雜的界線在哪裡？"),
        (13, "因成得敗、因敗得成如何理解？"),
        (15, "相神對全局有何作用？"),
        (17, "墓庫刑衝之說是否可信？"),
        (20, "生剋先後怎樣分吉凶？"),
        (25, "行運如何配合原局？"),
        (27, "喜忌在干支上有何分別？"),
        (31, "正官格局如何成立？"),
        (39, "七煞與偏官是否同一術語？"),
        (46, "建祿月劫如何取運？"),
    ]
    for index, (chapter, question) in enumerate(variant_specs, start=1):
        cases.append(
            case(
                f"zp-variant-{index:02d}",
                "traditional_variant",
                question,
                ids[chapter],
            )
        )

    citation_specs = [
        (8, "candidate-ziping-yongshen-month-command", "用神从月令讨论的候选规则引用了哪段原典？"),
        (15, "candidate-ziping-xiangshen-role", "相神候选卡的原文依据在哪里？"),
        (17, "", "墓库刑冲边界讨论怎样追溯原典？"),
        (21, "candidate-ziping-stars-boundary", "星辰无关格局的候选规则来源是什么？"),
        (25, "candidate-ziping-luck-context", "行运必须结合原局的候选卡引用何处？"),
        (8, "annotation:ziping-zhenquan:yongshen-scope", "用神术语范围注释绑定了哪些原文？"),
        (1, "annotation:ziping-zhenquan:provenance", "网络底本导入边界注释如何定位原文？"),
        (9, "", "请返回用神成败救应的可解析引用 ID 与原文。"),
        (13, "", "请定位因成得败一篇的原文引用。"),
        (16, "", "杂气取用的引用链应回到哪段原典？"),
        (18, "", "四吉神破格的结论应引用哪段原典？"),
        (19, "", "四凶神成格的结论应引用哪段原典？"),
        (20, "", "生克先后规则请给出原典引用。"),
        (27, "", "干支喜忌差别请给出可追溯原文。"),
        (47, "", "杂格取运讨论请返回原典引用而非图谱节点。"),
    ]
    for index, (chapter, target, question) in enumerate(citation_specs, start=1):
        cases.append(
            case(
                f"zp-citation-{index:02d}",
                "citation_chain",
                question,
                [*([target] if target else []), *ids[chapter]],
            )
        )

    expected_counts = {
        "chapter_location": 47,
        "concept_paraphrase": 30,
        "cross_chapter": 20,
        "prediction": 20,
        "traditional_variant": 18,
        "citation_chain": 15,
    }
    actual = {
        category: sum(item["category"] == category for item in cases)
        for category in expected_counts
    }
    if actual != expected_counts or len(cases) != 150:
        raise RuntimeError(f"Invalid evaluation distribution: {actual}, total={len(cases)}")
    return cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    corpus = yaml.safe_load(
        (root / "knowledge" / "originals" / "ziping-zhenquan.yml").read_text(encoding="utf-8")
    )
    cases = build(corpus)
    target = root / "evals" / "ziping-zhenquan.json"
    target.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {target}")


if __name__ == "__main__":
    main()
