#!/usr/bin/env python3
"""Archive and import explicitly selected public-domain Bazi chapters."""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BACKEND_ROOT / "knowledge/imports/classic-selections.yml"
GANZHI = re.compile(r"^[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]$")
NOTE_PREFIXES = ("眉批：", "眉批:", "【原注】", "**【原注】**", "原注：")
SITE_TEXT_MARKERS = ("白话译文", "现代启示", "关键词解释", "留给读者的问题")
USER_AGENT = "BaziKnowledgeImporter/2.0 (public-domain research archive)"


@dataclass(frozen=True)
class ParsedPage:
    title: str
    paragraphs: list[str]


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._article_depth = 0
        self._capture_tag = ""
        self._buffer: list[str] = []
        self.title = ""
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and attributes.get("id") == "article-content":
            self._article_depth = 1
            return
        if self._article_depth:
            if tag not in {"br", "img", "hr", "meta", "link", "input"}:
                self._article_depth += 1
            if tag in {"h1", "p"} and not self._capture_tag:
                self._capture_tag = tag
                self._buffer = []
            elif tag == "br" and self._capture_tag:
                self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._article_depth and self._capture_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._article_depth:
            return
        if tag == self._capture_tag:
            text = normalize("".join(self._buffer))
            if text:
                if tag == "h1" and not self.title:
                    self.title = text
                elif tag == "p":
                    self.paragraphs.append(text)
            self._capture_tag = ""
            self._buffer = []
        self._article_depth -= 1


def normalize(value: str) -> str:
    value = value.replace("\u3000", " ").replace("**", "")
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_page(data: bytes) -> ParsedPage:
    parser = ArticleParser()
    parser.feed(data.decode("utf-8"))
    if not parser.title or not parser.paragraphs:
        raise ValueError("source page lacks article#article-content")
    if any(marker in paragraph for marker in SITE_TEXT_MARKERS for paragraph in parser.paragraphs):
        raise ValueError("site-authored translation or insight leaked into original article")
    return ParsedPage(parser.title, parser.paragraphs)


def download(url: str, attempts: int = 4) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {url}")
                return response.read()
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1 + attempt)
    raise RuntimeError(f"download failed for {url}: {error}") from error


def is_pillars(text: str) -> bool:
    tokens = re.split(r"[、，,\s]+", text.strip())
    return len(tokens) == 4 and all(GANZHI.fullmatch(token) for token in tokens)


def is_luck_row(text: str) -> bool:
    tokens = re.split(r"[、，,\s]+", text.strip())
    return len(tokens) >= 3 and all(GANZHI.fullmatch(token) for token in tokens)


def pillar_tokens(text: str) -> list[str]:
    return re.split(r"[、，,\s]+", text.strip())


def clean_paragraphs(work: dict[str, object], page: ParsedPage) -> list[str]:
    title = page.title.strip()
    result: list[str] = []
    for paragraph in page.paragraphs:
        compact = re.sub(r"\s+", "", paragraph)
        if compact == re.sub(r"\s+", "", title):
            continue
        if re.fullmatch(r"第\s*[一二三四五六七八九十百零〇0-9]+\s*页", paragraph):
            continue
        if str(work["title"]) in paragraph and len(paragraph) <= 45:
            continue
        if result and result[-1] == paragraph:
            continue
        result.append(paragraph)
    return result


def classify_blocks(work: dict[str, object], page: ParsedPage) -> list[tuple[str, str, list[str]]]:
    """Return (layer, kind, paragraphs), preserving source order."""
    paragraphs = clean_paragraphs(work, page)
    blocks: list[tuple[str, str, list[str]]] = []
    index = 0
    ditiansui = work["parser"] == "ditiansui_commentary"
    while index < len(paragraphs):
        text = paragraphs[index]
        next_is_note = index + 1 < len(paragraphs) and paragraphs[index + 1].startswith(
            NOTE_PREFIXES
        )
        inline_note = re.search(r"[（(](眉批[：:].+?)[）)]\s*$", text)
        if inline_note:
            body = text[: inline_note.start()].rstrip()
            if body:
                blocks.append(("original", "text", [body]))
            blocks.append(("annotation", "original_note", [inline_note.group(1)]))
            index += 1
            continue
        if is_pillars(text) and index + 1 < len(paragraphs) and is_luck_row(paragraphs[index + 1]):
            case = [text, paragraphs[index + 1]]
            index += 2
            if index < len(paragraphs) and not is_pillars(paragraphs[index]):
                case.append(paragraphs[index])
                index += 1
            blocks.append(("original", "case", case))
            continue
        if ditiansui and next_is_note:
            blocks.append(("original", "text", [text]))
        elif text.startswith(NOTE_PREFIXES):
            blocks.append(("annotation", "original_note", [text]))
        elif ditiansui and re.match(r"^(?:【)?任氏曰(?:】)?[：:]?", text):
            blocks.append(("annotation", "ren_commentary", [text]))
        elif ditiansui and blocks and blocks[-1][1] in {"original_note", "ren_commentary"}:
            layer, kind, values = blocks[-1]
            blocks[-1] = (layer, kind, [*values, text])
        else:
            blocks.append(("original", "text", [text]))
        index += 1
    return blocks


def base_metadata(config: dict[str, object], source_url: str, content: str) -> dict[str, object]:
    defaults = dict(config["defaults"])
    return {
        **defaults,
        "content_sha256": digest_text(content),
        "source_pages": [source_url],
    }


def concepts_for(title: str, text: str) -> list[str]:
    terms = {
        "十神": ["十神", "正官", "七杀", "印绶", "财星", "食神", "伤官", "比肩", "劫财"],
        "旺衰": ["旺衰", "衰旺", "强弱", "身旺", "身弱"],
        "用神": ["用神", "体用", "病药"],
        "格局": ["格局", "成格", "破格"],
        "行运": ["大运", "流年", "岁运", "太岁", "行运"],
        "六亲": ["六亲", "父母", "兄弟", "夫妻", "子女", "婚姻"],
        "健康": ["疾病", "寿夭", "夭寿", "疾厄", "健康"],
        "性情": ["性情", "性格"],
        "调候": ["调候", "寒暖", "燥湿"],
    }
    haystack = title + text
    return [
        name for name, markers in terms.items() if any(marker in haystack for marker in markers)
    ]


def make_rule_card(
    work: dict[str, object], chapter: str, number: int, passage: dict[str, object]
) -> dict[str, object]:
    text = str(passage["text"])
    title = str(work["title"])
    health = "健康" in passage["concepts"]
    prohibited = [
        "不得把古籍断语表达为具体事件必然发生",
        "不得因一柱、一字或相似十神直接套用结论",
    ]
    if health:
        prohibited.extend(
            [
                "不得用于疾病诊断或推断用户患有何种疾病",
                "不得预测寿命、死亡年龄或灾祸必然发生",
                "不得提供治疗、用药或停药建议；有症状应咨询医生",
            ]
        )
    card_id = f"classic-rule-{work['slug']}-ch{chapter}-p{number:03d}"
    conclusion = text
    return {
        "id": card_id,
        "title": f"《{title}》{passage['chapter_path'][-1]}第{number}条",
        "content": text,
        "schema_version": 2,
        "card_type": "rule",
        "rule": f"《{title}》本条原文：{conclusion}",
        "school": work["school"],
        "concepts": passage["concepts"] or ["传统命理规则"],
        "premises": ["先核对完整四柱、月令与原局结构", "本卡只复述所引古籍的单条文献观点"],
        "conclusion": conclusion,
        "conditions": ["须结合原文所在章节语境", "涉及岁运时须同时比较原局与岁运"],
        "exceptions": ["原文未列明的适用边界不得自行补成确定规则"],
        "break_conditions": ["其他干支改变原文所述关系时，不得只凭局部套断"],
        "rescue_conditions": ["原文明确提及制化、救应或通变时应一并检索比较"],
        "priority": 55,
        "priority_note": "单一网页底本机器核验卡，低于人工校勘资料；跨书观点分别保留。",
        "counterexamples": ["仅出现相同十神但旺衰、位置或岁运条件不同，不构成同一案例"],
        "prohibited_uses": prohibited,
        "source_refs": [{"passage_id": passage["id"], "locator": passage["locator"]}],
        "graph_refs": passage["graph_refs"],
        **base_metadata(GLOBAL_CONFIG, str(passage["source_pages"][0]), text),
    }


def make_case_card(
    work: dict[str, object], chapter: str, number: int, passage: dict[str, object], rule_id: str
) -> dict[str, object]:
    lines = str(passage["text"]).splitlines()
    pillars = pillar_tokens(lines[0])
    text = str(passage["text"])
    prohibited = [
        "命例只展示原书的规则应用，不代表对现实用户的预测",
        "不得因一柱、一字或相似十神直接套用古人结果",
        "不得复述命例中的死亡、疾病、婚姻或财富结果为用户必然结局",
    ]
    if "健康" in passage["concepts"]:
        prohibited.extend(
            [
                "不得用于疾病诊断或推断用户患有何种疾病",
                "不得预测寿命、死亡年龄或灾祸必然发生",
                "不得提供治疗、用药或停药建议；有症状应咨询医生",
            ]
        )
    return {
        "id": f"classic-case-{work['slug']}-ch{chapter}-case{number:03d}",
        "title": f"《{work['title']}》{passage['chapter_path'][-1]}命例{number}",
        "content": text,
        "schema_version": 2,
        "card_type": "case",
        "school": work["school"],
        "concepts": passage["concepts"] or ["命例"],
        "conclusion": lines[-1],
        "prohibited_uses": prohibited,
        "source_refs": [{"passage_id": passage["id"], "locator": passage["locator"]}],
        "rule_refs": [rule_id],
        "case_pillars": pillars,
        "application_steps": ["核对四柱", "识别原文采用的结构条件", "对照原书所列行运与叙述结果"],
        "graph_refs": passage["graph_refs"],
        **base_metadata(GLOBAL_CONFIG, str(passage["source_pages"][0]), text),
    }


GLOBAL_CONFIG: dict[str, object] = {}


def import_work(
    config: dict[str, object], work: dict[str, object], refresh: bool
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    raw_root = BACKEND_ROOT / "knowledge/raw" / str(work["slug"])
    html_root = raw_root / "html"
    html_root.mkdir(parents=True, exist_ok=True)
    catalog_url = f"{config['source']['host']}/bazi/{work['catalog_id']}/"
    catalog_path = html_root / "catalog.html"
    if refresh or not catalog_path.exists():
        catalog_path.write_bytes(download(catalog_url))
    files = [{"path": "html/catalog.html", "url": catalog_url}]
    passages: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    cards: list[dict[str, object]] = []
    sequence = 0
    for chapter in work["chapters"]:
        source_url = f"{catalog_url}{chapter}/"
        path = html_root / f"chapter-{chapter}.html"
        if refresh or not path.exists():
            path.write_bytes(download(source_url))
        files.append({"path": f"html/chapter-{chapter}.html", "url": source_url})
        page = parse_page(path.read_bytes())
        blocks = classify_blocks(work, page)
        passage_number = annotation_number = case_number = 0
        last_rule_id = ""
        chapter_passages: list[dict[str, object]] = []
        for layer, kind, values in blocks:
            content = "\n".join(values)
            if layer == "annotation":
                annotation_number += 1
                related = [chapter_passages[-1]["id"]] if chapter_passages else []
                annotation_id = (
                    f"annotation-{work['slug']}-ch{chapter}-{kind}{annotation_number:03d}"
                )
                if kind == "original_note":
                    is_ditiansui = work["parser"] == "ditiansui_commentary"
                    annotation_label = "原注" if is_ditiansui else "眉批"
                    annotation_author = (
                        "刘基（传统题署）" if is_ditiansui else "原网页所录眉批（作者未详）"
                    )
                else:
                    annotation_label = "任铁樵注"
                    annotation_author = "任铁樵"
                annotations.append(
                    {
                        "id": annotation_id,
                        "title": f"《{work['title']}》{page.title}·{annotation_label}",
                        "kind": "commentary",
                        "content": content,
                        "passage_refs": related,
                        "author": annotation_author,
                        "publication": str(work["title"]),
                        "school": work["school"],
                        "concepts": concepts_for(page.title, content),
                        "graph_refs": [f"work:{work['slug']}", f"school:{work['slug']}"]
                        + (["person:ren-tieqiao"] if kind == "ren_commentary" else []),
                        **base_metadata(config, source_url, content),
                    }
                )
                continue
            sequence += 1
            passage_number += 1
            is_case = kind == "case"
            if is_case:
                case_number += 1
            passage_identifier = (
                f"{work['slug']}-ch{chapter}-case{case_number:03d}"
                if is_case
                else f"{work['slug']}-ch{chapter}-p{passage_number:03d}"
            )
            concepts = concepts_for(page.title, content)
            graph_refs = [f"work:{work['slug']}", f"school:{work['slug']}"]
            locator_item = f"命例{case_number}" if is_case else f"第{passage_number}条"
            passage = {
                "id": passage_identifier,
                "volume": "命例" if is_case else "选章正文",
                "chapter_path": [f"第{int(chapter)}章 {page.title}"],
                "sequence": sequence,
                "text": content,
                "normalized_text": re.sub(r"\s+", "", content),
                "locator": f"第{int(chapter)}章·{page.title}·{locator_item}",
                "concepts": concepts,
                "graph_refs": graph_refs,
                **base_metadata(config, source_url, content),
            }
            passages.append(passage)
            chapter_passages.append(passage)
            if is_case:
                if not last_rule_id:
                    synthetic = make_rule_card(work, chapter, passage_number, passage)
                    synthetic["card_type"] = "rule"
                    cards.append(synthetic)
                    last_rule_id = str(synthetic["id"])
                cards.append(make_case_card(work, chapter, case_number, passage, last_rule_id))
            else:
                rule = make_rule_card(work, chapter, passage_number, passage)
                cards.append(rule)
                last_rule_id = str(rule["id"])
    work_text = "\n".join(str(item["text"]) for item in passages)
    corpus = {
        "work": {
            "id": work["work_id"],
            "title": work["title"],
            "attributed_author": work["attributed_author"],
            "dynasty": work["dynasty"],
            "edition": work["edition"],
            "edition_notes": "按固定主题白名单选章；未作跨版本人工校勘。",
            "source_url": catalog_url,
            "rights": "public_domain",
            "school": work["school"],
            **base_metadata(config, catalog_url, work_text),
        },
        "passages": passages,
    }
    manifest_files = []
    for item in files:
        content = (raw_root / item["path"]).read_bytes()
        manifest_files.append(
            {**item, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest = {
        "source": {
            "catalog_url": catalog_url,
            "retrieved_at": config["source"]["retrieved_at"],
            "host": config["source"]["host"],
        },
        "selection": {
            "chapter_whitelist": work["chapters"],
            "chapter_count": len(work["chapters"]),
            "review_status": "machine_verified",
        },
        "parsing": {
            "included": "article#article-content only",
            "excluded": config["source"]["excluded_panels"],
            "layer_split": work["parser"],
        },
        "files": manifest_files,
        "counts": {"passages": len(passages), "annotations": len(annotations), "cards": len(cards)},
    }
    return corpus, annotations, cards, manifest


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_cards: list[dict[str, object]] = []
    for work in GLOBAL_CONFIG["works"]:
        corpus, annotations, cards, manifest = import_work(GLOBAL_CONFIG, work, args.refresh)
        slug = str(work["slug"])
        write_yaml(BACKEND_ROOT / "knowledge/originals" / f"{slug}.yml", corpus)
        write_yaml(
            BACKEND_ROOT / "knowledge/annotations" / f"{slug}.yml", {"annotations": annotations}
        )
        write_yaml(BACKEND_ROOT / "knowledge/raw" / slug / "manifest.yml", manifest)
        all_cards.extend(cards)
    write_yaml(BACKEND_ROOT / "knowledge/cards/classic-selections-v2.yml", {"cards": all_cards})
    print(f"Imported {len(GLOBAL_CONFIG['works'])} works and {len(all_cards)} cards")


if __name__ == "__main__":
    main()
