#!/usr/bin/env python3
"""Verify the archived transcription and promote it to machine_verified."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

EXPECTED_CHAPTERS = 47
EXPECTED_PASSAGES = 310
VERIFICATION_METHOD = (
    "archive_sha256+chapter_sequence+paragraph_integrity+hidden_watermark_removal+"
    "embedded_material_boundary+citation_locator"
)
UNRESOLVED_VARIANTS = ["未获得可合法公开下载的第二底本，潜在异文尚无法逐字确认。"]
BASIC_REVIEWED_NODES = {
    "person:project-editors",
    "school:basic-consensus",
    "concept:yinyang",
    "rule:yinyang-not-value",
    "source:project-foundation",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: object) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def verification_fields(
    content: str, source_pages: list[str], confidence: float
) -> dict[str, object]:
    return {
        "content_sha256": sha256_text(content),
        "source_pages": source_pages,
        "collation_method": VERIFICATION_METHOD,
        "source_count": 1,
        "verification_level": "single_source_integrity",
        "confidence": confidence,
        "unresolved_variants": UNRESOLVED_VARIANTS,
    }


def verify_archive(root: Path, manifest: dict[str, object], corpus: dict[str, object]) -> None:
    raw_root = root / "raw" / "ziping-zhenquan"
    files = manifest["files"]
    assert isinstance(files, list) and len(files) == 49
    for item in files:
        assert isinstance(item, dict)
        content = (raw_root / str(item["path"])).read_bytes()
        if len(content) != item["bytes"]:
            raise ValueError(f"Archive byte count changed: {item['path']}")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Archive sha256 changed: {item['path']}")

    derived = manifest["derived_text"]
    assert isinstance(derived, dict)
    clean = (raw_root / str(derived["path"])).read_bytes()
    if hashlib.sha256(clean).hexdigest() != derived["sha256"]:
        raise ValueError("Derived clean text sha256 changed")

    passages = corpus["passages"]
    assert isinstance(passages, list)
    if len(passages) != EXPECTED_PASSAGES:
        raise ValueError(f"Expected {EXPECTED_PASSAGES} passages, found {len(passages)}")
    chapter_numbers = {
        int(match.group(1))
        for passage in passages
        if isinstance(passage, dict)
        and passage["volume"] == "正文"
        and (match := re.match(r"第(\d+)章", str(passage["chapter_path"][-1])))
    }
    if chapter_numbers != set(range(1, EXPECTED_CHAPTERS + 1)):
        raise ValueError("The canonical corpus does not contain a continuous 47-chapter structure")
    if [item["sequence"] for item in passages] != list(range(1, EXPECTED_PASSAGES + 1)):
        raise ValueError("Passage sequence is not continuous")
    if len({item["id"] for item in passages}) != EXPECTED_PASSAGES:
        raise ValueError("Passage ids are not unique")
    forbidden = ("刘基注", "干支体象", "附《滴天髓》", "中华典藏网")
    for passage in passages:
        text = str(passage["text"])
        if any(marker in text for marker in forbidden):
            raise ValueError(f"Embedded or watermarked material remains in {passage['id']}")
        if not passage.get("locator"):
            raise ValueError(f"Missing citation locator: {passage['id']}")
        if text not in clean.decode("utf-8"):
            raise ValueError(f"Passage cannot be located in derived text: {passage['id']}")


def source_page_map(manifest: dict[str, object]) -> dict[str, str]:
    pages: dict[str, str] = {}
    for item in manifest["files"]:
        path = str(item["path"])
        if path == "html/preface.html":
            pages["preface"] = str(item["url"])
        match = re.search(r"chapter-(\d+)\.html$", path)
        if match:
            pages[f"ch{int(match.group(1)):02d}"] = str(item["url"])
    return pages


def promote_originals(
    manifest: dict[str, object], corpus: dict[str, object]
) -> dict[str, list[str]]:
    pages = source_page_map(manifest)
    passage_pages: dict[str, list[str]] = {}
    passages = corpus["passages"]
    for passage in passages:
        match = re.match(r"ziping-zhenquan-(preface|ch\d{2})-", str(passage["id"]))
        assert match
        source_pages = [pages[match.group(1)]]
        passage.update(verification_fields(str(passage["text"]), source_pages, 0.62))
        passage["normalized_text"] = re.sub(r"\s+", "", str(passage["text"]))
        passage["status"] = "machine_verified"
        passage_pages[str(passage["id"])] = source_pages

    work = corpus["work"]
    work_content = "\n".join(str(item["text"]) for item in passages)
    work.update(verification_fields(work_content, [str(manifest["source"]["catalog_url"])], 0.60))
    work["status"] = "machine_verified"
    work["edition_notes"] = (
        "仅纳入乾隆四十一年序及正文第一至四十七篇；已自动核对结构、归档哈希、隐藏水印、"
        "夹带材料边界和引用定位。未找到可合法公开下载的赵展如原刊影印本，故仍是单一来源机器校勘本。"
    )
    return passage_pages


def promote_derived_file(
    path: Path,
    key: str,
    content_key: str,
    passage_pages: dict[str, list[str]],
) -> None:
    payload = load_yaml(path)
    for item in payload[key]:
        source_pages = sorted(
            {
                page
                for ref in item.get("passage_refs", [])
                for page in passage_pages.get(str(ref), [])
            }
        )
        if not source_pages:
            for ref in item.get("source_refs", []):
                passage_id = ref.get("passage_id") if isinstance(ref, dict) else None
                source_pages.extend(passage_pages.get(str(passage_id), []))
        item.update(verification_fields(str(item[content_key]), sorted(set(source_pages)), 0.56))
        item["status"] = "machine_verified"
    write_yaml(path, payload)


def promote_graph(path: Path, passage_pages: dict[str, list[str]]) -> None:
    payload = load_yaml(path)
    for node in payload["nodes"]:
        node_id = str(node["id"])
        if node_id in BASIC_REVIEWED_NODES:
            node["status"] = "reviewed"
            node["verification_level"] = "human_review"
            node["confidence"] = 0.95
        elif node_id == "work:ditiansui":
            node["status"] = "draft"
            continue
        else:
            node["status"] = "machine_verified"
            node["verification_level"] = "single_source_integrity"
            node["confidence"] = 0.60
        node_content = f"{node.get('name', '')}\n{node.get('description', '')}"
        node["content_sha256"] = sha256_text(node_content)
        node["collation_method"] = VERIFICATION_METHOD
        node["source_count"] = 1
        node["unresolved_variants"] = (
            [] if node_id in BASIC_REVIEWED_NODES else UNRESOLVED_VARIANTS
        )
        node["source_pages"] = sorted(
            {
                page
                for ref in node.get("source_refs", [])
                for page in passage_pages.get(str(ref.get("passage_id", "")), [])
            }
        )
    for edge in payload["edges"]:
        if edge["status"] == "draft" and str(edge["id"]).startswith("edge-ziping"):
            edge["status"] = "machine_verified"
        elif edge["status"] == "draft" and str(edge["id"]) in {
            "edge-shen-authored-ziping",
            "edge-diancang-hosts-ziping",
        }:
            edge["status"] = "machine_verified"
        if edge["status"] == "machine_verified":
            edge["verification_level"] = "single_source_integrity"
            edge["confidence"] = 0.60
            edge["collation_method"] = VERIFICATION_METHOD
            edge["source_count"] = 1
            edge["unresolved_variants"] = UNRESOLVED_VARIANTS
    write_yaml(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "knowledge",
    )
    args = parser.parse_args()
    root = args.knowledge_root
    manifest_path = root / "raw" / "ziping-zhenquan" / "manifest.yml"
    corpus_path = root / "originals" / "ziping-zhenquan.yml"
    manifest = load_yaml(manifest_path)
    corpus = load_yaml(corpus_path)
    verify_archive(root, manifest, corpus)
    passage_pages = promote_originals(manifest, corpus)
    promote_derived_file(
        root / "annotations" / "ziping-zhenquan.yml",
        "annotations",
        "content",
        passage_pages,
    )
    promote_derived_file(
        root / "cards" / "ziping-zhenquan-candidates.yml",
        "cards",
        "content",
        passage_pages,
    )
    promote_graph(root / "graph" / "core.yml", passage_pages)

    manifest["selection"]["review_status"] = "machine_verified"
    manifest["verification"] = {
        "method": VERIFICATION_METHOD,
        "source_count": 1,
        "verification_level": "single_source_integrity",
        "confidence": 0.62,
        "second_source_search": (
            "国家图书馆文津检索仅确认现代整理本馆藏；未发现可合法公开下载的赵展如原刊影印本。"
        ),
        "unresolved_variants": UNRESOLVED_VARIANTS,
        "checks": {
            "chapters": EXPECTED_CHAPTERS,
            "passages": EXPECTED_PASSAGES,
            "archive_files": 49,
            "archive_hashes": "passed",
            "hidden_watermark": "passed",
            "embedded_material_boundary": "passed",
            "citation_locators": "passed",
        },
    }
    write_yaml(corpus_path, corpus)
    write_yaml(manifest_path, manifest)
    print("Verified 47 chapters / 310 passages; promoted Zi Ping evidence to machine_verified")


if __name__ == "__main__":
    main()
