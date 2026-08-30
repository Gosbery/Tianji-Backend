from __future__ import annotations

import re
from pathlib import Path

import yaml

from bazi_api.modules.retrieval.schemas import RetrievalDocument

from .schemas import KnowledgeCard, SourcePassage


class KnowledgeRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cards: list[KnowledgeCard] = []
        self.passages: list[SourcePassage] = []

    def load(self) -> None:
        self.cards = self._load_cards()
        self.passages = self._load_sources()
        if not self.cards:
            raise RuntimeError(f"No knowledge cards found under {self.root / 'cards'}")

    def _load_cards(self) -> list[KnowledgeCard]:
        cards: list[KnowledgeCard] = []
        for path in sorted((self.root / "cards").glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            items = payload.get("cards", []) if isinstance(payload, dict) else payload
            cards.extend(KnowledgeCard.model_validate(item) for item in items)
        return cards

    def _load_sources(self) -> list[SourcePassage]:
        passages: list[SourcePassage] = []
        for path in sorted((self.root / "sources").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            source_id = path.stem
            title = source_id
            chapter = "正文"
            buffer: list[str] = []
            index = 0

            def flush() -> None:
                nonlocal buffer, index
                body = "\n".join(buffer).strip()
                if not body:
                    return
                index += 1
                concepts = sorted({term for term in CONCEPT_TERMS if term in f"{chapter}{body}"})
                passages.append(
                    SourcePassage(
                        id=f"{source_id}:{index}",
                        source_id=source_id,
                        title=title,
                        chapter_path=chapter,
                        text=body,
                        locator=f"段落 {index}",
                        concepts=concepts,
                    )
                )
                buffer = []

            for line in text.splitlines():
                heading = re.match(r"^(#{1,4})\s+(.+)$", line)
                if heading:
                    flush()
                    chapter = heading.group(2).strip()
                    if len(heading.group(1)) == 1:
                        title = chapter
                    continue
                if line.strip() == "---":
                    continue
                buffer.append(line)
            flush()
        return passages

    def documents(self) -> list[RetrievalDocument]:
        docs: list[RetrievalDocument] = []
        for card in self.cards:
            if card.status != "reviewed":
                continue
            source = (
                ", ".join(f"{ref.source_id} · {ref.section}" for ref in card.source_refs)
                or "项目种子知识卡"
            )
            docs.append(
                RetrievalDocument(
                    id=card.id,
                    kind="knowledge_card",
                    title=card.title,
                    text=card.content,
                    source=source,
                    school=card.school,
                    concepts=card.concepts,
                    conditions=card.conditions,
                    exclusions=card.exclusions,
                )
            )
        for passage in self.passages:
            docs.append(
                RetrievalDocument(
                    id=passage.id,
                    kind="source_passage",
                    title=f"{passage.title} · {passage.chapter_path}",
                    text=passage.text,
                    source=f"{passage.source_id} · {passage.locator}",
                    school="基础共识",
                    concepts=passage.concepts,
                )
            )
        return docs


CONCEPT_TERMS = [
    "阴阳",
    "五行",
    "木",
    "火",
    "土",
    "金",
    "水",
    "甲",
    "乙",
    "丙",
    "丁",
    "戊",
    "己",
    "庚",
    "辛",
    "壬",
    "癸",
    "子",
    "丑",
    "寅",
    "卯",
    "辰",
    "巳",
    "午",
    "未",
    "申",
    "酉",
    "戌",
    "亥",
    "日主",
    "十神",
    "比肩",
    "劫财",
    "食神",
    "伤官",
    "偏财",
    "正财",
    "七杀",
    "正官",
    "偏印",
    "正印",
    "藏干",
    "月令",
    "旺衰",
    "合",
    "冲",
    "刑",
    "害",
    "流派",
]
