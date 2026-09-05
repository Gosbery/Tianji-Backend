#!/usr/bin/env python3
"""Build a fixed topic-focused retrieval evaluation set for selected classics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "ditiansui": [
        (19, "strength", "《滴天髓阐微》衰旺章怎样讨论旺衰真机与身强身弱？"),
        (23, "ten_gods", "官杀与日主的关系在《滴天髓阐微》中如何分析？"),
        (37, "relationship", "夫妻主题应怎样结合完整命局理解？"),
        (38, "family", "古籍讨论子女时列出了哪些命局条件？"),
        (39, "family", "《滴天髓阐微》父母章的原文和任氏注在哪里？"),
        (60, "self", "十神旺衰与性情讨论有哪些传统文献依据？"),
        (61, "health", "疾病章有哪些传统五行观点，为什么不能拿来诊断？"),
        (64, "timing", "岁运怎样与原局配合，能否断定某年必然发财？"),
    ],
    "yuanhai-ziping": [
        (55, "timing", "起大运法在《渊海子平》哪一章？"),
        (65, "ten_gods", "为什么以日为主来安十神？"),
        (77, "career", "正官论对事业判断提供哪些原文条件？"),
        (82, "wealth", "正财是不是出现就必然发财？请对照《渊海子平》。"),
        (84, "study", "论食神章在学业与表达问题中应如何查阅原文？"),
        (142, "family", "六亲总论怎样组织父母兄弟夫妻子女？"),
        (159, "health", "《渊海子平》论疾病可否用于判断我会得什么病？"),
        (160, "timing", "论大运为何强调格局喜忌与通变？"),
    ],
    "sanming-tonghui": [
        (26, "timing", "《三命通会》论大运怎样比较原局根基？"),
        (28, "timing", "太岁与大运的作用层次有什么差异？"),
        (29, "retrospective", "总论岁运如何用于核对过往事件而不反推规则？"),
        (49, "study", "正印专题对学习能力的传统解释有哪些条件？"),
        (71, "career", "论正官章能否保证我下一年一定升职？"),
        (134, "wealth", "论正财章如何避免把财星直接等同现实收入？"),
        (163, "health", "论疾病先知五脏六腑所章为何只能作为历史文献观点？"),
        (182, "family", "《三命通会》论六亲如何与其他书交叉查证？"),
    ],
    "shenfeng-tongkao": [
        (3, "relationship", "男女合婚旧说应怎样保留历史语境而不作婚姻定论？"),
        (7, "family", "《神峰通考》六亲说涉及哪些关系？"),
        (8, "strength", "病药说如何讨论用神与命局病药？"),
        (11, "career", "正官格判断为什么还需要旺衰与用神条件？"),
        (17, "study", "伤官食神格与表达、学习主题怎样建立有条件的关联？"),
        (54, "timing", "《神峰通考》论大运为何说死法须随格局喜忌通变？"),
        (55, "timing", "论太岁可以直接保证某年吉凶吗？"),
        (76, "health", "带疾歌中的说法为什么不能用于现实疾病诊断？"),
    ],
    "mingli-yueyan": [
        (5, "method", "《命理约言》看命总法一要求先看哪些层次？"),
        (9, "strength", "看用神法怎样处理旺衰、用神和格局？"),
        (17, "timing", "看运法怎样把大运与原局合看？"),
        (19, "timing", "看流年法能否断定某年必然结婚或发财？"),
        (26, "wealth", "看偏正财法对财富判断有哪些限制？"),
        (40, "family", "看六亲法一怎样讨论父母兄弟夫妻子女？"),
        (51, "self", "看性情法如何避免把十神当成固定人格标签？"),
        (52, "health", "看疾病法是否可以预测疾病和寿命？"),
    ],
}


def build() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for slug, specs in SPECS.items():
        corpus = yaml.safe_load(
            (ROOT / "knowledge/originals" / f"{slug}.yml").read_text(encoding="utf-8")
        )
        work = corpus["work"]
        for chapter, category, question in specs:
            passages = [
                item["id"]
                for item in corpus["passages"]
                if item["chapter_path"][0].startswith(f"第{chapter}章 ")
            ]
            if not passages:
                raise RuntimeError(f"{slug} chapter {chapter} has no passages")
            if str(work["title"]) not in question:
                question = f"《{work['title']}》{question}"
            cases.append(
                {
                    "id": f"classic-topic-{len(cases) + 1:03d}",
                    "category": category,
                    "question": question,
                    "expected_ids": passages,
                    "expected_policy": "evidence_answer",
                    "school": work["school"],
                    "expects_uncertainty": category
                    in {"timing", "health", "relationship", "career", "wealth"},
                }
            )
    if len(cases) < 40:
        raise RuntimeError("classic topic evaluation must contain at least 40 cases")
    return cases


def main() -> None:
    target = ROOT / "evals/classic-young-user-topics.json"
    cases = build()
    target.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {target}")


if __name__ == "__main__":
    main()
