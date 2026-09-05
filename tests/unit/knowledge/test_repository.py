import hashlib
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.knowledge.schemas import GraphEdge, GraphNode, KnowledgeCard, SourceRef

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def _minimal_knowledge_root(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    (root / "cards").mkdir(parents=True)
    (root / "graph").mkdir()
    (root / "sources").mkdir()
    (root / "cards" / "test.yml").write_text(
        yaml.safe_dump(
            {
                "review_defaults": {
                    "verification_level": "human_review",
                    "reviewed_by": "test-reviewer",
                    "reviewed_at": "2026-09-03",
                    "review_note": "测试夹具人工确认",
                },
                "cards": [
                    {
                        "id": "test-yinyang",
                        "title": "阴阳边界",
                        "content": "阴阳不是价值判断。",
                        "concepts": ["阴阳"],
                        "status": "reviewed",
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "graph" / "core.yml").write_text(
        yaml.safe_dump(
            {
                "review_defaults": {
                    "verification_level": "human_review",
                    "reviewed_by": "test-reviewer",
                    "reviewed_at": "2026-09-03",
                    "review_note": "测试夹具人工确认",
                },
                "nodes": [
                    {
                        "id": "concept:yinyang",
                        "type": "concept",
                        "name": "阴阳",
                        "aliases": ["阴阳关系"],
                        "status": "reviewed",
                    }
                ],
                "edges": [],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "sources" / "notes.md").write_text(
        "---\n"
        "reviewed_by: test-reviewer\n"
        "reviewed_at: 2026-09-03\n"
        "review_note: 测试夹具人工确认\n"
        "---\n\n"
        "# 研究笔记\n\n## 阴阳关系\n\n这里讨论阴阳关系。\n",
        encoding="utf-8",
    )
    return root


def test_seed_knowledge_has_four_validated_layers() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()

    overview = repository.overview()
    assert repository.cards
    assert repository.original_passages
    assert repository.annotations
    assert repository.graph_nodes
    assert repository.graph_edges
    assert overview.layers["knowledge_cards"] == len(repository.cards)
    assert overview.layers["canonical_passages"] == len(repository.original_passages)
    assert overview.layers["modern_annotations"] == len(repository.annotations)
    assert overview.layers["graph_nodes"] == len(repository.graph_nodes)
    assert overview.layers["graph_edges"] == len(repository.graph_edges)
    assert overview.preview_documents >= overview.retrieval_documents > 0
    assert overview.retrieval_documents == len(repository.documents())


def test_only_reviewed_evidence_enters_retrieval() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()

    documents = repository.documents()
    kinds = {document.kind for document in documents}

    assert "canonical_passage" not in kinds
    assert {"modern_annotation", "knowledge_card"}.issubset(kinds)
    assert all(document.layer in {1, 2, 3} for document in documents)
    yinyang = next(document for document in documents if document.id == "concept-yinyang")
    assert "禁用范围" in yinyang.text
    assert "concept:yinyang" in yinyang.graph_refs
    assert any(ref.startswith("annotation:project-foundation:") for ref in yinyang.trace_refs)
    assert not any(document.id.startswith("ziping-zhenquan-") for document in documents)
    assert not any(document.id.startswith("candidate-ziping-") for document in documents)

    preview = repository.documents("personal_preview")
    assert any(document.id.startswith("ziping-zhenquan-") for document in preview)
    assert any(document.id.startswith("candidate-ziping-") for document in preview)
    assert not any(document.review_status in {"draft", "retired"} for document in preview)
    assert all(
        document.warning for document in preview if document.review_status == "machine_verified"
    )
    generated = next(document for document in preview if document.id.startswith("ziping-rule"))
    card = next(item for item in repository.cards if item.id == generated.id)
    assert card.conclusion in generated.rerank_text
    assert card.conditions[0] in generated.rerank_text
    assert card.exceptions[0] in generated.rerank_text
    assert card.break_conditions[0] in generated.rerank_text
    assert card.prohibited_uses[0] in generated.rerank_text


def test_reviewed_status_requires_complete_human_audit_metadata() -> None:
    with pytest.raises(ValidationError, match="missing audit fields"):
        GraphNode(id="concept:test", type="concept", name="测试", status="reviewed")

    reviewed = GraphNode(
        id="concept:test",
        type="concept",
        name="测试",
        status="reviewed",
        verification_level="human_review",
        reviewed_by="reviewer",
        reviewed_at="2026-09-03",
        review_note="已核对来源和内容",
    )

    assert reviewed.verification_level == "human_review"
    assert reviewed.reviewed_by == "reviewer"


def test_ziping_source_archive_is_complete_and_hash_verified() -> None:
    raw_root = BACKEND_ROOT / "knowledge" / "raw" / "ziping-zhenquan"
    manifest = yaml.safe_load((raw_root / "manifest.yml").read_text(encoding="utf-8"))
    corpus = yaml.safe_load(
        (BACKEND_ROOT / "knowledge" / "originals" / "ziping-zhenquan.yml").read_text(
            encoding="utf-8"
        )
    )

    passages = corpus["passages"]
    chapter_numbers = {
        int(match.group(1))
        for passage in passages
        if passage["volume"] == "正文"
        and (match := re.match(r"第(\d+)章", passage["chapter_path"][-1]))
    }

    assert corpus["work"]["status"] == "machine_verified"
    assert chapter_numbers == set(range(1, 48))
    assert len(passages) == 310
    assert len({passage["id"] for passage in passages}) == len(passages)
    assert [passage["sequence"] for passage in passages] == list(range(1, len(passages) + 1))
    assert {passage["status"] for passage in passages} == {"machine_verified"}
    assert all(len(passage["content_sha256"]) == 64 for passage in passages)
    assert all(passage["verification_level"] == "single_source_integrity" for passage in passages)
    assert all(passage["source_count"] == 1 for passage in passages)
    assert not any(
        "刘基注" in passage["text"] or "干支体象" in passage["text"] for passage in passages
    )

    assert manifest["selection"]["chapter_count"] == 47
    assert manifest["selection"]["review_status"] == "machine_verified"
    assert manifest["verification"]["checks"]["archive_hashes"] == "passed"
    assert manifest["verification"]["checks"]["citation_locators"] == "passed"
    assert manifest["source"]["txt_download"]["status"] == "login_required_not_downloaded"
    assert len(manifest["files"]) == 49
    for item in manifest["files"]:
        source_path = raw_root / item["path"]
        content = source_path.read_bytes()
        assert len(content) == item["bytes"]
        assert hashlib.sha256(content).hexdigest() == item["sha256"]

    derived = manifest["derived_text"]
    clean_text = (raw_root / derived["path"]).read_bytes()
    assert len(clean_text) == derived["bytes"]
    assert hashlib.sha256(clean_text).hexdigest() == derived["sha256"]


def test_ziping_v2_cards_cover_all_chapters_with_explicit_reasoning_fields() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()
    cards = [
        card
        for card in repository.cards
        if card.schema_version == 2 and card.id.startswith("ziping-")
    ]
    rules = [card for card in cards if card.card_type != "case"]
    cases = [card for card in cards if card.card_type == "case"]
    chapters = {
        int(match.group(1)) for card in rules if (match := re.search(r"-ch(\d+)-", card.id))
    }

    assert 150 <= len(cards) <= 250
    assert len(rules) == 231
    assert len(cases) == 12
    assert chapters == set(range(1, 48))
    assert sum(f"-ch{chapter:02d}-" in card.id for chapter in range(8, 21) for card in rules) >= 71
    assert (
        sum(f"-ch{chapter:02d}-" in card.id for chapter in range(31, 48) for card in rules) >= 143
    )
    assert all(card.status == "machine_verified" for card in cards)
    assert len({re.sub(r"\s+", "", card.rule) for card in rules}) == len(rules)
    assert all(not card.conclusion.rstrip().endswith(("？", "?")) for card in rules)
    assert all(card.conclusion in card.rule for card in rules)
    assert all(
        card.premises
        and card.conclusion
        and card.conditions
        and card.exceptions
        and card.break_conditions
        and card.rescue_conditions
        and card.priority_note
        and card.counterexamples
        and any(ref.passage_id for ref in card.source_refs)
        for card in rules
    )
    assert all(
        len(card.case_pillars) == 4
        and card.application_steps
        and card.rule_refs
        and card.conclusion
        for card in cases
    )


def test_v2_rule_card_rejects_incomplete_reasoning_shape() -> None:
    with pytest.raises(ValidationError, match="v2 rule card missing"):
        KnowledgeCard(
            id="incomplete-v2",
            title="不完整规则",
            content="只有摘要",
            schema_version=2,
            card_type="rule",
            concepts=["格局"],
        )


def test_graph_edges_must_reference_existing_nodes() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()
    repository.graph_edges.append(
        GraphEdge(
            id="broken-edge",
            source="concept:yinyang",
            target="missing:node",
            relation="references",
        )
    )

    with pytest.raises(ValueError, match="unknown graph node"):
        repository._validate()


def test_reviewed_and_machine_status_dependencies_are_enforced() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()
    candidate = next(card for card in repository.cards if card.id.startswith("candidate-ziping-"))
    candidate.status = "reviewed"

    with pytest.raises(ValueError, match="reviewed.*machine_verified"):
        repository._validate()

    candidate.status = "machine_verified"
    draft_target = repository.original_passages[0]
    draft_target.status = "draft"
    candidate.source_refs.append(SourceRef(passage_id=draft_target.id))
    with pytest.raises(ValueError, match="machine_verified.*draft"):
        repository._validate()


def test_overview_is_cached_after_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = KnowledgeRepository(_minimal_knowledge_root(tmp_path))
    repository.load()
    expected = repository.overview()

    def fail_if_documents_are_rebuilt(*args: object, **kwargs: object) -> None:
        raise AssertionError("overview must not rebuild retrieval documents")

    monkeypatch.setattr(repository, "documents", fail_if_documents_are_rebuilt)

    assert repository.overview() == expected


def test_legacy_concepts_are_derived_from_graph_nodes(tmp_path: Path) -> None:
    repository = KnowledgeRepository(_minimal_knowledge_root(tmp_path))
    repository.load()

    note = next(item for item in repository.annotations if item.id.startswith("annotation:notes:"))

    assert note.concepts == ["阴阳"]
    assert set(note.concepts) <= {
        node.name for node in repository.graph_nodes if node.type == "concept"
    }


def test_topic_and_bibliography_catalogs_are_loaded_and_validated() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()

    assert len(repository.topics) == 12
    assert {item.priority for item in repository.topics} == {
        "core",
        "high_frequency",
        "extended",
    }
    health = next(item for item in repository.topics if item.id == "topic:health")
    assert health.risk == "health"
    assert {"疾病", "非诊断"}.issubset(health.retrieval_terms)
    assert len(repository.bibliography) >= 7
    qianli = next(
        item
        for item in repository.bibliography
        if item.id == "bibliography:wei-qianli-qianli-minggao"
    )
    assert qianli.rights_status == "copyrighted"
    assert not qianli.fulltext_eligible
    assert all(
        item.rights_status == "public_domain"
        for item in repository.bibliography
        if item.fulltext_eligible
    )


def test_classic_preview_evidence_never_enters_reviewed_only_scope() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()

    reviewed_ids = {item.id for item in repository.documents("reviewed_only")}
    preview = repository.documents("personal_preview")

    assert not any(item.startswith("classic-rule-") for item in reviewed_ids)
    assert any(item.id.startswith("classic-rule-") for item in preview)
    assert all(
        item.warning for item in preview if item.id.startswith(("classic-rule-", "classic-case-"))
    )


def test_classic_health_and_prediction_cards_preserve_safety_boundaries() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()
    classics = [item for item in repository.cards if item.id.startswith("classic-")]
    health = [item for item in classics if "健康" in item.concepts]
    cases = [item for item in classics if item.card_type == "case"]

    assert health
    assert all(any("疾病诊断" in value for value in item.prohibited_uses) for item in health)
    assert all(any("寿命" in value for value in item.prohibited_uses) for item in health)
    assert all(
        any("必然" in value for value in item.prohibited_uses)
        for item in classics
        if item.card_type == "rule"
    )
    assert all(any("直接套用" in value for value in item.prohibited_uses) for item in cases)
