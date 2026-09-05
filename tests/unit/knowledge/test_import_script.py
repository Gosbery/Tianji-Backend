import hashlib
import re
import runpy
from pathlib import Path
from typing import Any

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def test_importer_derives_terms_and_refs_from_graph() -> None:
    namespace: dict[str, Any] = runpy.run_path(
        str(BACKEND_ROOT / "scripts" / "import_ziping_zhenquan.py")
    )
    catalog = namespace["load_graph_concepts"](BACKEND_ROOT / "knowledge" / "graph")

    concepts = namespace["concepts_for"]("论用神", "以月令取用", catalog)
    refs = namespace["graph_refs_for"](concepts, catalog)

    assert {"月令", "用神"}.issubset(concepts)
    assert "concept:month-command" in refs
    assert "concept:yongshen" in refs
    assert "work:ziping-zhenquan" in refs


def test_classic_import_config_uses_explicit_whitelists() -> None:
    config = yaml.safe_load(
        (BACKEND_ROOT / "knowledge/imports/classic-selections.yml").read_text(encoding="utf-8")
    )

    assert len(config["works"]) == 5
    assert {item["work_id"] for item in config["works"]} == {
        "work-ditiansui",
        "work-yuanhai-ziping",
        "work-sanming-tonghui",
        "work-shenfeng-tongkao",
        "work-mingli-yueyan",
    }
    assert all(
        item["chapters"] and len(item["chapters"]) == len(set(item["chapters"]))
        for item in config["works"]
    )
    assert config["source"]["excluded_panels"] == ["白话译文", "关键词", "现代启示"]


def test_classic_archives_and_generated_corpora_are_integrity_checked() -> None:
    config = yaml.safe_load(
        (BACKEND_ROOT / "knowledge/imports/classic-selections.yml").read_text(encoding="utf-8")
    )
    for work in config["works"]:
        slug = work["slug"]
        raw_root = BACKEND_ROOT / "knowledge/raw" / slug
        manifest = yaml.safe_load((raw_root / "manifest.yml").read_text(encoding="utf-8"))
        corpus = yaml.safe_load(
            (BACKEND_ROOT / "knowledge/originals" / f"{slug}.yml").read_text(encoding="utf-8")
        )
        passages = corpus["passages"]

        assert manifest["selection"]["chapter_whitelist"] == work["chapters"]
        assert manifest["selection"]["chapter_count"] == len(work["chapters"])
        assert manifest["parsing"]["included"] == "article#article-content only"
        assert [item["sequence"] for item in passages] == list(range(1, len(passages) + 1))
        assert len({item["id"] for item in passages}) == len(passages)
        assert all(item["status"] == "machine_verified" for item in passages)
        assert all(item["verification_level"] == "single_source_integrity" for item in passages)
        assert all(item["confidence"] <= 0.65 for item in passages)
        for item in manifest["files"]:
            content = (raw_root / item["path"]).read_bytes()
            assert len(content) == item["bytes"]
            assert hashlib.sha256(content).hexdigest() == item["sha256"]


def test_classic_layer_split_excludes_site_copy_and_explicit_notes() -> None:
    originals = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (BACKEND_ROOT / "knowledge/originals").glob("*.yml")
        if path.stem
        in {"ditiansui", "yuanhai-ziping", "sanming-tonghui", "shenfeng-tongkao", "mingli-yueyan"}
    )
    ditiansui_annotations = yaml.safe_load(
        (BACKEND_ROOT / "knowledge/annotations/ditiansui.yml").read_text(encoding="utf-8")
    )["annotations"]

    assert not any(marker in originals for marker in ("白话译文", "现代启示", "关键词解释"))
    assert "【原注】" not in originals
    assert "任氏曰" not in originals
    assert "眉批：" not in originals
    assert {item["author"] for item in ditiansui_annotations} == {
        "刘基（传统题署）",
        "任铁樵",
    }
    assert all(item["passage_refs"] for item in ditiansui_annotations)


def test_classic_case_cards_have_stable_ids_and_complete_pillars() -> None:
    cards = yaml.safe_load(
        (BACKEND_ROOT / "knowledge/cards/classic-selections-v2.yml").read_text(encoding="utf-8")
    )["cards"]
    cases = [item for item in cards if item["card_type"] == "case"]

    assert cases
    assert all(
        re.fullmatch(r"classic-case-[a-z-]+-ch\d{3}-case\d{3}", item["id"]) for item in cases
    )
    assert all(len(item["case_pillars"]) == 4 for item in cases)
    assert all(item["rule_refs"] and item["application_steps"] for item in cases)
    assert all(any("不得因一柱" in value for value in item["prohibited_uses"]) for item in cases)
