#!/usr/bin/env python3
"""Verify selected-classic archives, generated layers, and repository integrity."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bazi_api.modules.knowledge.repository import KnowledgeRepository  # noqa: E402

DEFAULT_CONFIG = ROOT / "knowledge/imports/classic-selections.yml"
FORBIDDEN_ORIGINAL_MARKERS = (
    "白话译文",
    "现代启示",
    "关键词解释",
    "【原注】",
    "任氏曰",
    "眉批：",
)


def verify(config_path: Path, knowledge_root: Path) -> dict[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    totals = {"works": 0, "passages": 0, "annotations": 0, "cards": 0, "cases": 0}
    card_payload = yaml.safe_load(
        (knowledge_root / "cards/classic-selections-v2.yml").read_text(encoding="utf-8")
    )
    cards = card_payload["cards"]
    card_ids = {item["id"] for item in cards}
    if len(card_ids) != len(cards):
        raise ValueError("classic cards contain duplicate IDs")

    for work in config["works"]:
        slug = work["slug"]
        raw_root = knowledge_root / "raw" / slug
        manifest = yaml.safe_load((raw_root / "manifest.yml").read_text(encoding="utf-8"))
        corpus = yaml.safe_load(
            (knowledge_root / "originals" / f"{slug}.yml").read_text(encoding="utf-8")
        )
        annotations = yaml.safe_load(
            (knowledge_root / "annotations" / f"{slug}.yml").read_text(encoding="utf-8")
        )["annotations"]
        passages = corpus["passages"]
        if manifest["selection"]["chapter_whitelist"] != work["chapters"]:
            raise ValueError(f"{slug} manifest whitelist differs from import config")
        if [item["sequence"] for item in passages] != list(range(1, len(passages) + 1)):
            raise ValueError(f"{slug} passage sequence is not continuous")
        if len({item["id"] for item in passages}) != len(passages):
            raise ValueError(f"{slug} contains duplicate passage IDs")
        for item in manifest["files"]:
            content = (raw_root / item["path"]).read_bytes()
            if len(content) != item["bytes"]:
                raise ValueError(f"{slug}/{item['path']} byte count mismatch")
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"{slug}/{item['path']} SHA-256 mismatch")
        for passage in passages:
            if passage["status"] != "machine_verified" or passage["confidence"] > 0.65:
                raise ValueError(f"{passage['id']} has invalid preview trust metadata")
            if any(marker in passage["text"] for marker in FORBIDDEN_ORIGINAL_MARKERS):
                raise ValueError(f"{passage['id']} contains excluded editorial content")
        totals["works"] += 1
        totals["passages"] += len(passages)
        totals["annotations"] += len(annotations)
        totals["cases"] += sum(item["volume"] == "命例" for item in passages)

    classic_cards = [item for item in cards if item["id"].startswith("classic-")]
    for card in classic_cards:
        if card["card_type"] == "case":
            if len(card["case_pillars"]) != 4:
                raise ValueError(f"{card['id']} does not contain four pillars")
            if not re.fullmatch(r"classic-case-[a-z-]+-ch\d{3}-case\d{3}", card["id"]):
                raise ValueError(f"{card['id']} is not a stable case ID")
        if not card["source_refs"] or not card["source_refs"][0].get("passage_id"):
            raise ValueError(f"{card['id']} lacks a canonical passage reference")
    totals["cards"] = len(classic_cards)

    repository = KnowledgeRepository(knowledge_root)
    repository.load()
    if len(repository.topics) < 12 or len(repository.bibliography) < 7:
        raise ValueError("topic or bibliography catalog is incomplete")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--knowledge-root", type=Path, default=ROOT / "knowledge")
    args = parser.parse_args()
    totals = verify(args.config, args.knowledge_root)
    print(
        "Classic knowledge verified: "
        + ", ".join(f"{key}={value}" for key, value in totals.items())
    )


if __name__ == "__main__":
    main()
