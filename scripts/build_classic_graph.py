#!/usr/bin/env python3
"""Build graph nodes and traceable relationships for selected classics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "knowledge/imports/classic-selections.yml"
TOPICS = ROOT / "knowledge/catalog/young-user-topics.yml"
OUTPUT = ROOT / "knowledge/graph/classic-selections.yml"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata(content: str, pages: list[str] | None = None) -> dict[str, object]:
    return {
        "status": "machine_verified",
        "verification_level": "single_source_integrity",
        "confidence": 0.6,
        "content_sha256": digest(content),
        "collation_method": "catalog_mapping+source_passage_refs+stable_graph_ids",
        "source_count": 1,
        "unresolved_variants": ["图谱关系随单一网页选章建立，尚待人工复核。"],
        "source_pages": pages or [],
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    topic_catalog = yaml.safe_load(TOPICS.read_text(encoding="utf-8"))
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    work_graph_by_id: dict[str, str] = {}
    first_passage_by_work: dict[str, str] = {}

    author_ids = {
        "ditiansui": ("person:ren-tieqiao", "任铁樵"),
        "yuanhai-ziping": ("person:xu-dasheng", "徐大升"),
        "sanming-tonghui": ("person:wan-minying", "万民英"),
        "shenfeng-tongkao": ("person:zhang-nan", "张楠"),
        "mingli-yueyan": ("person:chen-suan", "陈素庵"),
    }
    for person_id, name in dict(author_ids.values()).items():
        description = f"{name}与首批经典选章的传统作者或注者关系。"
        nodes.append(
            {
                "id": person_id,
                "type": "person",
                "name": name,
                "description": description,
                **metadata(description),
            }
        )

    original_description = "《滴天髓》原文层，用于与任铁樵《滴天髓阐微》的注本关系分层。"
    nodes.append(
        {
            "id": "work:ditiansui-original",
            "type": "work",
            "name": "滴天髓",
            "description": original_description,
            **metadata(original_description),
        }
    )

    for work in config["works"]:
        slug = work["slug"]
        corpus = yaml.safe_load(
            (ROOT / "knowledge/originals" / f"{slug}.yml").read_text(encoding="utf-8")
        )
        first_passage = corpus["passages"][0]
        first_passage_by_work[work["work_id"]] = first_passage["id"]
        work_graph_id = f"work:{slug}"
        school_graph_id = f"school:{slug}"
        work_graph_by_id[work["work_id"]] = work_graph_id
        description = f"{work['title']}首批主题选章，当前为单一公开网页底本的个人预览资料。"
        nodes.append(
            {
                "id": work_graph_id,
                "type": "work",
                "name": work["title"],
                "description": description,
                "school": work["school"],
                "source_refs": [{"passage_id": first_passage["id"]}],
                **metadata(description, first_passage["source_pages"]),
            }
        )
        school_description = f"本项目用于区分《{work['title']}》选章观点的来源流派标签。"
        nodes.append(
            {
                "id": school_graph_id,
                "type": "school",
                "name": work["school"],
                "description": school_description,
                **metadata(school_description),
            }
        )
        author_id, _ = author_ids[slug]
        edge_description = f"{author_id}与《{work['title']}》的传统题署或注者关系。"
        edges.append(
            {
                "id": f"edge-{slug}-author",
                "source": author_id,
                "target": work_graph_id,
                "relation": "comments_on" if slug == "ditiansui" else "attributed_author_of",
                "description": edge_description,
                "source_refs": [{"passage_id": first_passage["id"]}],
                **metadata(edge_description, first_passage["source_pages"]),
            }
        )
        school_edge_description = f"《{work['title']}》选章按{work['school']}标签组织召回。"
        edges.append(
            {
                "id": f"edge-{slug}-school",
                "source": work_graph_id,
                "target": school_graph_id,
                "relation": "foundational_for",
                "description": school_edge_description,
                "source_refs": [{"passage_id": first_passage["id"]}],
                **metadata(school_edge_description, first_passage["source_pages"]),
            }
        )

    dts_description = "《滴天髓阐微》按页面标记拆出原文、原注与任铁樵注。"
    edges.append(
        {
            "id": "edge-ditiansui-chanwei-comments-on-original",
            "source": "work:ditiansui",
            "target": "work:ditiansui-original",
            "relation": "comments_on",
            "description": dts_description,
            "source_refs": [{"passage_id": first_passage_by_work["work-ditiansui"]}],
            **metadata(dts_description),
        }
    )

    topic_by_label: dict[str, str] = {}
    for topic in topic_catalog["topics"]:
        graph_id = topic["id"]
        topic_by_label[topic["label"]] = graph_id
        nodes.append(
            {
                "id": graph_id,
                "type": "topic",
                "name": topic["label"],
                "aliases": topic["query_terms"],
                "description": topic["description"],
                **metadata(topic["description"]),
            }
        )
        for work_id in topic["related_work_ids"]:
            graph_work = work_graph_by_id.get(work_id, "work:ziping-zhenquan")
            passage_id = first_passage_by_work.get(work_id, "ziping-zhenquan-ch01-p001")
            description = f"{graph_work}覆盖主题“{topic['label']}”；具体结论仍按原文逐条引用。"
            edges.append(
                {
                    "id": f"edge-{graph_work.split(':', 1)[1]}-covers-{graph_id.split(':', 1)[1]}",
                    "source": graph_work,
                    "target": graph_id,
                    "relation": "covers",
                    "description": description,
                    "source_refs": [{"passage_id": passage_id}],
                    **metadata(description),
                }
            )

    cards = yaml.safe_load(
        (ROOT / "knowledge/cards/classic-selections-v2.yml").read_text(encoding="utf-8")
    )["cards"]
    concept_topics = {
        "六亲": "topic:family",
        "健康": "topic:health",
        "行运": "topic:luck-timing",
        "旺衰": "topic:strength-climate",
        "调候": "topic:strength-climate",
        "十神": "topic:ten-gods-self",
        "性情": "topic:ten-gods-self",
    }
    for card in cards:
        if card["card_type"] != "case":
            continue
        node_id = f"card:{card['id']}"
        description = "四柱、行运与原书叙述已单列，禁止直接套用于现实用户。"
        nodes.append(
            {
                "id": node_id,
                "type": "knowledge_card",
                "name": card["title"],
                "description": description,
                "source_refs": card["source_refs"],
                **metadata(description, card["source_pages"]),
            }
        )
        targets = {concept_topics[item] for item in card["concepts"] if item in concept_topics}
        if not targets:
            targets = {"topic:retrospective"}
        for target in sorted(targets):
            edge_description = f"该命例展示“{target}”相关规则在原书中的应用。"
            edges.append(
                {
                    "id": f"edge-{card['id']}-illustrates-{target.split(':', 1)[1]}",
                    "source": node_id,
                    "target": target,
                    "relation": "illustrates",
                    "description": edge_description,
                    "source_refs": card["source_refs"],
                    **metadata(edge_description, card["source_pages"]),
                }
            )

    OUTPUT.write_text(
        yaml.safe_dump(
            {"nodes": nodes, "edges": edges}, allow_unicode=True, sort_keys=False, width=120
        ),
        encoding="utf-8",
    )
    print(f"Built {len(nodes)} nodes and {len(edges)} edges")


if __name__ == "__main__":
    main()
