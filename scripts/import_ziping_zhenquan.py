#!/usr/bin/env python3
"""Download and import the public web transcription of Zi Ping Zhen Quan."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import yaml

CATALOG_URL = "https://www.diancangwang.cn/xuanxuewushu/03cf53eede8a/"
WORK_ID = "work-ziping-zhenquan"
EXPECTED_CHAPTERS = 47
USER_AGENT = "BaziKnowledgeImporter/1.0 (source archival for research)"
PAGE_SUFFIX_MARKERS = {2: "附《滴天髓》之七天干篇。"}
DEFAULT_GRAPH_ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "graph"


@dataclass(frozen=True)
class CatalogEntry:
    title: str
    url: str


@dataclass(frozen=True)
class GraphConcept:
    id: str
    name: str
    terms: tuple[str, ...]


class CatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[CatalogEntry] = []
        self._booklist_depth = 0
        self._anchor_href = ""
        self._anchor_title = ""
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "div":
            if self._booklist_depth:
                self._booklist_depth += 1
            elif attributes.get("id") == "booklist":
                self._booklist_depth = 1
        if tag == "a" and self._booklist_depth:
            self._anchor_href = attributes.get("href") or ""
            self._anchor_title = attributes.get("title") or ""
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor_href:
            title = normalize_space(self._anchor_title or "".join(self._anchor_text))
            self.entries.append(CatalogEntry(title=title, url=self._anchor_href))
            self._anchor_href = ""
            self._anchor_title = ""
            self._anchor_text = []
        if tag == "div" and self._booklist_depth:
            self._booklist_depth -= 1


class ChapterParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.paragraphs: list[str] = []
        self._in_h1 = False
        self._pageview_depth = 0
        self._in_paragraph = False
        self._paragraph_text: list[str] = []
        self._hidden_span_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h1" and not self._pageview_depth:
            self._in_h1 = True
        if tag == "div":
            if self._pageview_depth:
                self._pageview_depth += 1
            elif attributes.get("id") == "pageview":
                self._pageview_depth = 1
        if tag == "p" and self._pageview_depth:
            self._in_paragraph = True
            self._paragraph_text = []
        if tag == "span" and "display:none" in (attributes.get("style") or "").replace(" ", ""):
            self._hidden_span_depth += 1

    def handle_data(self, data: str) -> None:
        if self._in_h1:
            self.title += data
        if self._in_paragraph and not self._hidden_span_depth:
            self._paragraph_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self._in_h1 = False
            self.title = normalize_space(self.title)
        if tag == "span" and self._hidden_span_depth:
            self._hidden_span_depth -= 1
        if tag == "p" and self._in_paragraph:
            paragraph = normalize_space("".join(self._paragraph_text))
            if paragraph:
                self.paragraphs.append(paragraph)
            self._in_paragraph = False
            self._paragraph_text = []
        if tag == "div" and self._pageview_depth:
            self._pageview_depth -= 1


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u3000", " ")).strip()


def chinese_number(value: str) -> int:
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "十" in value:
        tens, ones = value.split("十", 1)
        return (digits.get(tens, 1) * 10) + digits.get(ones, 0)
    return digits[value]


def parse_chapter_number(title: str) -> int | None:
    match = re.match(r"^([一二三四五六七八九十]+)(?:[．、.]|\s*\(附\))", title)
    return chinese_number(match.group(1)) if match else None


def download(url: str, attempts: int = 5) -> bytes:
    error: Exception | None = None
    alternatives = [url]
    if url.startswith("https://www."):
        alternatives.extend(
            [
                url.replace("https://www.", "https://", 1),
                url.replace("https://", "http://", 1),
            ]
        )
    for attempt in range(attempts):
        try:
            target = alternatives[attempt % len(alternatives)]
            request = Request(target, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=25) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {target}")
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retries need the original network error
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2 + attempt * 2)
    raise RuntimeError(f"Failed to download {url}: {error}") from error


def load_or_download(path: Path, url: str, refresh: bool) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    data = download(url)
    path.write_bytes(data)
    return data


def parse_catalog(raw_html: bytes) -> list[CatalogEntry]:
    parser = CatalogParser()
    parser.feed(raw_html.decode("utf-8"))
    return [CatalogEntry(entry.title, urljoin(CATALOG_URL, entry.url)) for entry in parser.entries]


def parse_page(raw_html: bytes) -> ChapterParser:
    parser = ChapterParser()
    parser.feed(raw_html.decode("utf-8"))
    if not parser.title or not parser.paragraphs:
        raise ValueError("Source page is missing its title or body")
    return parser


def select_original_paragraphs(chapter_number: int | None, page: ChapterParser) -> None:
    marker = PAGE_SUFFIX_MARKERS.get(chapter_number)
    if marker is None:
        return
    try:
        marker_index = page.paragraphs.index(marker)
    except ValueError as exc:
        message = f"Expected editorial addendum marker is missing from {page.title}"
        raise ValueError(message) from exc
    page.paragraphs = page.paragraphs[:marker_index]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chapter_basename(chapter_number: int | None) -> str:
    return "preface" if chapter_number is None else f"chapter-{chapter_number:02d}"


def passage_id(chapter_number: int | None, paragraph_number: int) -> str:
    section = "preface" if chapter_number is None else f"ch{chapter_number:02d}"
    return f"ziping-zhenquan-{section}-p{paragraph_number:03d}"


def load_graph_concepts(graph_root: Path) -> list[GraphConcept]:
    concepts: list[GraphConcept] = []
    seen_ids: set[str] = set()
    for path in sorted(graph_root.glob("*.y*ml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for node in payload.get("nodes", []):
            if node.get("type") != "concept" or node.get("status") == "retired":
                continue
            concept_id = str(node["id"])
            if concept_id in seen_ids:
                raise ValueError(f"Duplicate graph concept id: {concept_id}")
            seen_ids.add(concept_id)
            name = str(node["name"])
            concepts.append(
                GraphConcept(
                    id=concept_id,
                    name=name,
                    terms=tuple(dict.fromkeys([name, *map(str, node.get("aliases", []))])),
                )
            )
    if not concepts:
        raise ValueError(f"No concept nodes found under {graph_root}")
    return concepts


def concepts_for(
    title: str, text: str, catalog: list[GraphConcept]
) -> list[str]:
    haystack = f"{title}{text}"
    return [
        concept.name
        for concept in catalog
        if any(term in haystack for term in concept.terms)
    ]


def graph_refs_for(concepts: list[str], catalog: list[GraphConcept]) -> list[str]:
    graph_id_by_name = {concept.name: concept.id for concept in catalog}
    refs = {"work:ziping-zhenquan", "school:ziping-pattern"}
    refs.update(graph_id_by_name[concept] for concept in concepts)
    return sorted(refs)


def build_corpus(
    pages: list[tuple[int | None, ChapterParser]],
    graph_root: Path = DEFAULT_GRAPH_ROOT,
) -> dict[str, object]:
    concept_catalog = load_graph_concepts(graph_root)
    passages: list[dict[str, object]] = []
    sequence = 0
    for chapter_number, page in pages:
        section_title = "序" if chapter_number is None else re.sub(
            r"^[一二三四五六七八九十]+[．、.]\s*", "", page.title
        )
        chapter_label = "序" if chapter_number is None else f"第{chapter_number}章 {section_title}"
        for paragraph_number, text in enumerate(page.paragraphs, start=1):
            sequence += 1
            concepts = concepts_for(section_title, text, concept_catalog)
            passages.append(
                {
                    "id": passage_id(chapter_number, paragraph_number),
                    "volume": "序" if chapter_number is None else "正文",
                    "chapter_path": ["序"] if chapter_number is None else ["正文", chapter_label],
                    "sequence": sequence,
                    "text": text,
                    "locator": f"{chapter_label}·第{paragraph_number}段",
                    "concepts": concepts,
                    "graph_refs": graph_refs_for(concepts, concept_catalog),
                    "status": "draft",
                }
            )
    return {
        "work": {
            "id": WORK_ID,
            "title": "子平真诠",
            "attributed_author": "沈孝瞻（序称沈燡燔）",
            "dynasty": "清",
            "edition": "中华典藏网公开录入本（网页未标明具体刊本）",
            "edition_notes": (
                "来源目录收有清乾隆四十一年序、徐乐吾凡例、正文四十七篇及徐乐吾附篇。"
                "本次只纳入序与正文第一至四十七篇；移除网页隐藏水印并归一化空白，"
                "未作文字订正，尚待与馆藏《耕寸集》或赵展如原刊影印本逐字校勘。"
            ),
            "source_url": CATALOG_URL,
            "rights": "public_domain",
            "status": "draft",
        },
        "passages": passages,
    }


def build_clean_text(pages: list[tuple[int | None, ChapterParser]]) -> str:
    parts = ["子平真诠", "沈孝瞻（传统归属）", ""]
    for _, page in pages:
        parts.extend([page.title, "", *page.paragraphs, ""])
    return "\n".join(parts).rstrip() + "\n"


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "knowledge",
    )
    parser.add_argument("--retrieved-on", default=date.today().isoformat())
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download pages again instead of resuming from the local raw archive.",
    )
    args = parser.parse_args()

    raw_root = args.knowledge_root / "raw" / "ziping-zhenquan"
    html_root = raw_root / "html"
    html_root.mkdir(parents=True, exist_ok=True)

    catalog_html = load_or_download(html_root / "catalog.html", CATALOG_URL, args.refresh)
    entries = parse_catalog(catalog_html)
    preface = next((entry for entry in entries if entry.title == "序"), None)
    chapters = sorted(
        (
            (number, entry)
            for entry in entries
            if (number := parse_chapter_number(entry.title)) is not None
            and 1 <= number <= EXPECTED_CHAPTERS
        ),
        key=lambda item: item[0],
    )
    if preface is None:
        raise ValueError("Catalog is missing the preface")
    if [number for number, _ in chapters] != list(range(1, EXPECTED_CHAPTERS + 1)):
        raise ValueError("Catalog does not contain the expected continuous 47 chapters")

    selected = [(None, preface), *chapters]
    pages: list[tuple[int | None, ChapterParser]] = []
    source_files: list[dict[str, object]] = [
        {
            "path": "html/catalog.html",
            "url": CATALOG_URL,
            "sha256": sha256(catalog_html),
            "bytes": len(catalog_html),
        }
    ]
    for chapter_number, entry in selected:
        filename = f"{chapter_basename(chapter_number)}.html"
        raw_html = load_or_download(html_root / filename, entry.url, args.refresh)
        page = parse_page(raw_html)
        if page.title != entry.title:
            raise ValueError(f"Title mismatch for {entry.url}: {entry.title!r} != {page.title!r}")
        select_original_paragraphs(chapter_number, page)
        pages.append((chapter_number, page))
        source_files.append(
            {
                "path": f"html/{filename}",
                "url": entry.url,
                "title": entry.title,
                "sha256": sha256(raw_html),
                "bytes": len(raw_html),
            }
        )

    clean_text = build_clean_text(pages).encode("utf-8")
    clean_text_path = raw_root / "ziping-zhenquan.txt"
    clean_text_path.write_bytes(clean_text)
    corpus = build_corpus(pages, args.knowledge_root / "graph")
    write_yaml(args.knowledge_root / "originals" / "ziping-zhenquan.yml", corpus)

    excluded = []
    for entry in entries:
        if entry.title == "凡例":
            excluded.append(
                {
                    "title": entry.title,
                    "url": entry.url,
                    "reason": "徐乐吾为《子平真诠评注》所作凡例，不属于沈孝瞻清代原著。",
                }
            )
        elif parse_chapter_number(entry.title) == 48:
            excluded.append(
                {
                    "title": entry.title,
                    "url": entry.url,
                    "reason": "目录标为附篇，正文内容属于徐乐吾补充，不作为清代原著导入。",
                }
            )
    excluded.append(
        {
            "title": "第二篇网页所附《滴天髓》与《渊海子平》材料",
            "url": next(entry.url for number, entry in chapters if number == 2),
            "reason": "网页把其他典籍材料附在本篇之后；原始网页保留，派生正文从附文标记处截断。",
        }
    )
    manifest = {
        "source": {
            "title": "子平真诠",
            "attributed_author": "沈孝瞻",
            "provider": "中华典藏网",
            "catalog_url": CATALOG_URL,
            "retrieved_on": args.retrieved_on,
            "access": "无需登录的公开在线阅读页",
            "provider_notice": "网站页脚标注为非营利站点，仅供学习。",
            "txt_download": {
                "url": (
                    "https://www.diancangwang.cn/e/DownSys/DownSoft/"
                    "?classid=8&id=11386&pathid=0"
                ),
                "status": "login_required_not_downloaded",
            },
        },
        "selection": {
            "included": "乾隆四十一年序及正文第一至四十七篇",
            "chapter_count": EXPECTED_CHAPTERS,
            "excluded": excluded,
            "review_status": "draft",
            "normalization": "移除 display:none 隐藏水印，解码 HTML 实体并归一化空白；未订正文句。",
        },
        "derived_text": {
            "path": clean_text_path.name,
            "sha256": sha256(clean_text),
            "bytes": len(clean_text),
        },
        "files": source_files,
    }
    write_yaml(raw_root / "manifest.yml", manifest)

    passage_count = len(corpus["passages"])
    print(f"Imported {EXPECTED_CHAPTERS} chapters and {passage_count} passages")
    print(f"Raw source: {raw_root}")


if __name__ == "__main__":
    main()
