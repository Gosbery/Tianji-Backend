from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import yaml

from bazi_api.modules.retrieval.schemas import RetrievalDocument

from .schemas import (
    BibliographyCatalog,
    BibliographyEntry,
    CanonicalPassage,
    CanonicalWork,
    EvidenceScope,
    GraphEdge,
    GraphNode,
    KnowledgeCard,
    KnowledgeOverview,
    KnowledgeTopic,
    ModernAnnotation,
    SourceRef,
    TopicCatalog,
)


class KnowledgeRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.works: list[CanonicalWork] = []
        self.original_passages: list[CanonicalPassage] = []
        self.annotations: list[ModernAnnotation] = []
        self.cards: list[KnowledgeCard] = []
        self.graph_nodes: list[GraphNode] = []
        self.graph_edges: list[GraphEdge] = []
        self.topics: list[KnowledgeTopic] = []
        self.bibliography: list[BibliographyEntry] = []
        self._overview: KnowledgeOverview | None = None

    def load(self) -> None:
        self._overview = None
        self.works, self.original_passages = self._load_originals()
        self.cards = self._load_cards()
        self.graph_nodes, self.graph_edges = self._load_graph()
        self.annotations = [*self._load_annotations(), *self._load_legacy_sources()]
        self.topics = self._load_topics()
        self.bibliography = self._load_bibliography()
        if not self.cards:
            raise RuntimeError(f"No knowledge cards found under {self.root / 'cards'}")
        self._validate()
        self._overview = self._calculate_overview()

    def _load_topics(self) -> list[KnowledgeTopic]:
        path = self.root / "catalog" / "young-user-topics.yml"
        if not path.exists():
            return []
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return TopicCatalog.model_validate(payload).topics

    def _load_bibliography(self) -> list[BibliographyEntry]:
        path = self.root / "catalog" / "later-commentaries.yml"
        if not path.exists():
            return []
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BibliographyCatalog.model_validate(payload).entries

    def _load_originals(self) -> tuple[list[CanonicalWork], list[CanonicalPassage]]:
        works: list[CanonicalWork] = []
        passages: list[CanonicalPassage] = []
        for path in sorted((self.root / "originals").glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            work = CanonicalWork.model_validate(payload["work"])
            works.append(work)
            for item in payload.get("passages", []):
                passages.append(CanonicalPassage.model_validate({"work_id": work.id, **item}))
        return works, passages

    def _load_annotations(self) -> list[ModernAnnotation]:
        annotations: list[ModernAnnotation] = []
        for path in sorted((self.root / "annotations").glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            items = payload.get("annotations", []) if isinstance(payload, dict) else payload
            annotations.extend(ModernAnnotation.model_validate(item) for item in items)
        return annotations

    def _load_cards(self) -> list[KnowledgeCard]:
        cards: list[KnowledgeCard] = []
        for path in sorted((self.root / "cards").glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            items = payload.get("cards", []) if isinstance(payload, dict) else payload
            defaults = payload.get("review_defaults", {}) if isinstance(payload, dict) else {}
            cards.extend(KnowledgeCard.model_validate({**defaults, **item}) for item in items)
        return cards

    def _load_graph(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for path in sorted((self.root / "graph").glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            defaults = payload.get("review_defaults", {})
            nodes.extend(
                GraphNode.model_validate(
                    {**defaults, **item} if item.get("status") == "reviewed" else item
                )
                for item in payload.get("nodes", [])
            )
            edges.extend(
                GraphEdge.model_validate(
                    {**defaults, **item} if item.get("status") == "reviewed" else item
                )
                for item in payload.get("edges", [])
            )
        return nodes, edges

    def _load_legacy_sources(self) -> list[ModernAnnotation]:
        """Treat existing Markdown research files as layer-2 research notes."""

        annotations: list[ModernAnnotation] = []
        for path in sorted((self.root / "sources").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            review_metadata: dict[str, object] = {}
            if text.startswith("---\n"):
                _, frontmatter, text = text.split("---\n", 2)
                raw_metadata = yaml.safe_load(frontmatter) or {}
                review_metadata = {
                    key: raw_metadata[key]
                    for key in ("reviewed_by", "reviewed_at", "review_note")
                    if key in raw_metadata
                }
            is_reviewed = all(
                review_metadata.get(key) for key in ("reviewed_by", "reviewed_at", "review_note")
            )
            source_id = path.stem
            document_title = source_id
            headings: list[str] = []
            buffer: list[str] = []
            index = 0

            def flush() -> None:
                nonlocal buffer, index
                body = "\n".join(buffer).strip()
                if not body:
                    return
                index += 1
                section = " / ".join(headings) or "正文"
                concepts = self._extract_graph_concepts(f"{section}{body}")
                annotations.append(
                    ModernAnnotation(
                        id=f"annotation:{source_id}:{index}",
                        title=f"{document_title} · {section}",
                        kind="research_note",
                        content=body,
                        publication=document_title,
                        concepts=concepts,
                        source_refs=[
                            SourceRef(
                                source_id=source_id,
                                section=headings[-1] if headings else section,
                                locator=f"段落 {index}",
                            )
                        ],
                        status="reviewed" if is_reviewed else "draft",
                        verification_level="human_review" if is_reviewed else "unverified",
                        **review_metadata,
                    )
                )
                buffer = []

            for line in text.splitlines():
                heading = re.match(r"^(#{1,4})\s+(.+)$", line)
                if heading:
                    flush()
                    level = len(heading.group(1))
                    heading_text = heading.group(2).strip()
                    if level == 1:
                        document_title = heading_text
                    headings[:] = headings[: max(0, level - 1)]
                    headings.append(heading_text)
                    continue
                if line.strip() == "---":
                    continue
                buffer.append(line)
            flush()
        return annotations

    def _validate(self) -> None:
        self._require_unique("work", (item.id for item in self.works))
        self._require_unique("canonical passage", (item.id for item in self.original_passages))
        self._require_unique("annotation", (item.id for item in self.annotations))
        self._require_unique("knowledge card", (item.id for item in self.cards))
        self._require_unique("graph node", (item.id for item in self.graph_nodes))
        self._require_unique("graph edge", (item.id for item in self.graph_edges))
        self._require_unique("topic", (item.id for item in self.topics))
        self._require_unique("bibliography entry", (item.id for item in self.bibliography))

        work_ids = {item.id for item in self.works}
        work_status = {item.id: item.status for item in self.works}
        passage_ids = {item.id for item in self.original_passages}
        passage_status = {item.id: item.status for item in self.original_passages}
        annotation_ids = {item.id for item in self.annotations}
        annotation_status = {item.id: item.status for item in self.annotations}
        card_by_id = {item.id: item for item in self.cards}
        node_ids = {item.id for item in self.graph_nodes}
        node_status = {item.id: item.status for item in self.graph_nodes}
        source_ids = {
            ref.source_id
            for annotation in self.annotations
            for ref in annotation.source_refs
            if ref.source_id
        }
        for topic in self.topics:
            if not topic.query_terms or not topic.retrieval_terms:
                raise ValueError(f"{topic.id} requires query_terms and retrieval_terms")
            for work_id in topic.related_work_ids:
                self._require_reference("work", work_id, work_ids, topic.id)
        for entry in self.bibliography:
            if entry.source_work_id:
                self._require_reference("work", entry.source_work_id, work_ids, entry.id)

        for passage in self.original_passages:
            self._require_reference("work", passage.work_id, work_ids, passage.id)
            self._require_status_dependency(
                passage.status, work_status[passage.work_id], passage.id, passage.work_id
            )
            self._validate_content_hash(passage.id, passage.text, passage.content_sha256)
            self._require_graph_refs(passage.graph_refs, node_ids, passage.id)
        for annotation in self.annotations:
            for passage_id in annotation.passage_refs:
                self._require_reference("passage", passage_id, passage_ids, annotation.id)
                self._require_status_dependency(
                    annotation.status, passage_status[passage_id], annotation.id, passage_id
                )
            self._validate_content_hash(
                annotation.id, annotation.content, annotation.content_sha256
            )
            self._validate_source_refs(
                annotation.source_refs, source_ids, passage_ids, annotation_ids, annotation.id
            )
            self._validate_ref_statuses(
                annotation.status,
                annotation.source_refs,
                passage_status,
                annotation_status,
                annotation.id,
            )
            self._require_graph_refs(annotation.graph_refs, node_ids, annotation.id)
        for card in self.cards:
            for annotation_id in card.annotation_refs:
                self._require_reference("annotation", annotation_id, annotation_ids, card.id)
                self._require_status_dependency(
                    card.status, annotation_status[annotation_id], card.id, annotation_id
                )
            self._validate_content_hash(card.id, card.content, card.content_sha256)
            self._validate_source_refs(
                card.source_refs, source_ids, passage_ids, annotation_ids, card.id
            )
            self._validate_ref_statuses(
                card.status, card.source_refs, passage_status, annotation_status, card.id
            )
            self._require_graph_refs(card.graph_refs, node_ids, card.id)
            self._validate_card_refs(card, card_by_id)
        generated_rules: dict[str, str] = {}
        for card in self.cards:
            if "explicit_reasoning_fields" not in card.collation_method or card.card_type == "case":
                continue
            normalized_rule = re.sub(r"\s+", "", card.rule)
            duplicate = generated_rules.get(normalized_rule)
            if duplicate is not None:
                raise ValueError(f"{card.id} duplicates generated rule from {duplicate}")
            generated_rules[normalized_rule] = card.id
        for node in self.graph_nodes:
            self._validate_source_refs(
                node.source_refs, source_ids, passage_ids, annotation_ids, node.id
            )
            self._validate_ref_statuses(
                node.status, node.source_refs, passage_status, annotation_status, node.id
            )
        for edge in self.graph_edges:
            self._require_reference("graph node", edge.source, node_ids, edge.id)
            self._require_reference("graph node", edge.target, node_ids, edge.id)
            self._require_status_dependency(
                edge.status, node_status[edge.source], edge.id, edge.source
            )
            self._require_status_dependency(
                edge.status, node_status[edge.target], edge.id, edge.target
            )
            self._validate_source_refs(
                edge.source_refs, source_ids, passage_ids, annotation_ids, edge.id
            )
            self._validate_ref_statuses(
                edge.status, edge.source_refs, passage_status, annotation_status, edge.id
            )

        passages_by_work: defaultdict[str, list[CanonicalPassage]] = defaultdict(list)
        for passage in self.original_passages:
            passages_by_work[passage.work_id].append(passage)
        for work in self.works:
            ordered = sorted(passages_by_work[work.id], key=lambda item: item.sequence)
            self._validate_content_hash(
                work.id, "\n".join(item.text for item in ordered), work.content_sha256
            )

    @staticmethod
    def _require_unique(kind: str, identifiers: Iterable[str]) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for identifier in identifiers:
            if identifier in seen:
                duplicates.add(identifier)
            seen.add(identifier)
        if duplicates:
            raise ValueError(f"Duplicate {kind} ids: {', '.join(sorted(duplicates))}")

    @staticmethod
    def _require_reference(kind: str, target: str, known: set[str], owner: str) -> None:
        if target not in known:
            raise ValueError(f"{owner} references unknown {kind}: {target}")

    @classmethod
    def _validate_source_refs(
        cls,
        refs: Iterable[SourceRef],
        source_ids: set[str],
        passage_ids: set[str],
        annotation_ids: set[str],
        owner: str,
    ) -> None:
        for ref in refs:
            if ref.source_id:
                cls._require_reference("source", ref.source_id, source_ids, owner)
            if ref.passage_id:
                cls._require_reference("passage", ref.passage_id, passage_ids, owner)
            if ref.annotation_id:
                cls._require_reference("annotation", ref.annotation_id, annotation_ids, owner)

    @classmethod
    def _require_graph_refs(cls, refs: Iterable[str], node_ids: set[str], owner: str) -> None:
        for node_id in refs:
            cls._require_reference("graph node", node_id, node_ids, owner)

    @staticmethod
    def _require_status_dependency(
        owner_status: str, target_status: str, owner: str, target: str
    ) -> None:
        allowed = {
            "reviewed": {"reviewed"},
            "machine_verified": {"machine_verified", "reviewed"},
        }
        if owner_status in allowed and target_status not in allowed[owner_status]:
            raise ValueError(f"{owner} ({owner_status}) cannot cite {target} ({target_status})")

    @classmethod
    def _validate_ref_statuses(
        cls,
        owner_status: str,
        refs: Iterable[SourceRef],
        passage_status: dict[str, str],
        annotation_status: dict[str, str],
        owner: str,
    ) -> None:
        for ref in refs:
            if ref.passage_id:
                cls._require_status_dependency(
                    owner_status, passage_status[ref.passage_id], owner, ref.passage_id
                )
            if ref.annotation_id:
                cls._require_status_dependency(
                    owner_status, annotation_status[ref.annotation_id], owner, ref.annotation_id
                )

    @staticmethod
    def _validate_content_hash(owner: str, content: str, expected: str) -> None:
        if not expected:
            return
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(f"{owner} content_sha256 mismatch: {expected} != {actual}")

    @classmethod
    def _validate_card_refs(cls, card: KnowledgeCard, cards: dict[str, KnowledgeCard]) -> None:
        for ref_id in card.rule_refs:
            cls._require_reference("knowledge card", ref_id, set(cards), card.id)
            target = cards[ref_id]
            if target.card_type not in {"rule", "method", "boundary", "dispute"}:
                raise ValueError(f"{card.id} rule_refs non-rule card: {ref_id}")
            cls._require_status_dependency(card.status, target.status, card.id, ref_id)
        for ref_id in card.case_refs:
            cls._require_reference("knowledge card", ref_id, set(cards), card.id)
            target = cards[ref_id]
            if target.card_type != "case":
                raise ValueError(f"{card.id} case_refs non-case card: {ref_id}")
            cls._require_status_dependency(card.status, target.status, card.id, ref_id)

    def overview(self) -> KnowledgeOverview:
        if self._overview is None:
            self._overview = self._calculate_overview()
        return self._overview.model_copy(deep=True)

    def _calculate_overview(self) -> KnowledgeOverview:
        layers = {
            "canonical_passages": len(self.original_passages),
            "modern_annotations": len(self.annotations),
            "knowledge_cards": len(self.cards),
            "graph_nodes": len(self.graph_nodes),
            "graph_edges": len(self.graph_edges),
        }
        reviewed = {
            "canonical_passages": sum(item.status == "reviewed" for item in self.original_passages),
            "modern_annotations": sum(item.status == "reviewed" for item in self.annotations),
            "knowledge_cards": sum(item.status == "reviewed" for item in self.cards),
            "graph_nodes": sum(item.status == "reviewed" for item in self.graph_nodes),
            "graph_edges": sum(item.status == "reviewed" for item in self.graph_edges),
        }
        machine_verified = {
            "canonical_passages": sum(
                item.status == "machine_verified" for item in self.original_passages
            ),
            "modern_annotations": sum(
                item.status == "machine_verified" for item in self.annotations
            ),
            "knowledge_cards": sum(item.status == "machine_verified" for item in self.cards),
            "graph_nodes": sum(item.status == "machine_verified" for item in self.graph_nodes),
            "graph_edges": sum(item.status == "machine_verified" for item in self.graph_edges),
        }
        schools: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"reviewed": 0, "machine_verified": 0}
        )
        for document in self.documents("personal_preview"):
            if document.review_status in {"reviewed", "machine_verified"}:
                schools[document.school][document.review_status] += 1
        return KnowledgeOverview(
            layers=layers,
            reviewed=reviewed,
            machine_verified=machine_verified,
            retrieval_documents=len(self.documents()),
            preview_documents=len(self.documents("personal_preview")),
            schools=dict(sorted(schools.items())),
        )

    def _extract_graph_concepts(self, text: str) -> list[str]:
        return sorted(
            {
                node.name
                for node in self.graph_nodes
                if node.type == "concept"
                and any(term and term in text for term in [node.name, *node.aliases])
            }
        )

    def documents(self, scope: EvidenceScope = "reviewed_only") -> list[RetrievalDocument]:
        docs: list[RetrievalDocument] = []
        works = {item.id: item for item in self.works}
        passages = {item.id: item for item in self.original_passages}

        for passage in self.original_passages:
            if not self._status_allowed(passage.status, scope):
                continue
            work = works[passage.work_id]
            graph_refs, graph_terms = self._graph_context(
                passage.graph_refs, passage.concepts, scope
            )
            docs.append(
                RetrievalDocument(
                    id=passage.id,
                    kind="canonical_passage",
                    layer=1,
                    title=f"{work.title} · {' / '.join(passage.chapter_path)}",
                    text=passage.text,
                    source=f"{work.title}（{work.edition}）· {passage.locator}",
                    school=work.school,
                    concepts=passage.concepts,
                    trace_refs=[passage.id, passage.work_id],
                    graph_refs=graph_refs,
                    retrieval_terms=graph_terms,
                    normalized_text=passage.normalized_text,
                    review_status=passage.status,
                    verification_level=passage.verification_level,
                    confidence=passage.confidence,
                    warning=self._warning(passage.status, passage.unresolved_variants),
                    unresolved_variants=passage.unresolved_variants,
                    content_sha256=passage.content_sha256 or self._content_sha256(passage.text),
                )
            )

        for annotation in self.annotations:
            if not self._status_allowed(annotation.status, scope):
                continue
            graph_refs, graph_terms = self._graph_context(
                annotation.graph_refs, annotation.concepts, scope
            )
            passage_citations = [
                f"{works[passages[item].work_id].title} · {passages[item].locator}"
                for item in annotation.passage_refs
            ]
            source = "；".join(passage_citations) or self._format_refs(annotation.source_refs)
            docs.append(
                RetrievalDocument(
                    id=annotation.id,
                    kind="modern_annotation",
                    layer=2,
                    title=annotation.title,
                    text=annotation.content,
                    source=source or annotation.publication or "现代注释",
                    school=annotation.school,
                    concepts=annotation.concepts,
                    trace_refs=[annotation.id, *annotation.passage_refs],
                    graph_refs=graph_refs,
                    retrieval_terms=graph_terms,
                    review_status=annotation.status,
                    verification_level=annotation.verification_level,
                    confidence=annotation.confidence,
                    warning=self._warning(annotation.status, annotation.unresolved_variants),
                    unresolved_variants=annotation.unresolved_variants,
                    content_sha256=annotation.content_sha256
                    or self._content_sha256(annotation.content),
                    version=annotation.version,
                )
            )

        for card in self.cards:
            if not self._status_allowed(card.status, scope):
                continue
            graph_refs, graph_terms = self._graph_context(card.graph_refs, card.concepts, scope)
            text_parts = [card.content]
            if card.rule:
                text_parts.append(f"规则：{card.rule}")
            if card.premises:
                text_parts.append(f"前提：{'；'.join(card.premises)}")
            if card.conditions:
                text_parts.append(f"条件：{'；'.join(card.conditions)}")
            if card.conclusion:
                text_parts.append(f"结论：{card.conclusion}")
            if card.exceptions:
                text_parts.append(f"例外：{'；'.join(card.exceptions)}")
            if card.break_conditions:
                text_parts.append(f"破格条件：{'；'.join(card.break_conditions)}")
            if card.rescue_conditions:
                text_parts.append(f"救应条件：{'；'.join(card.rescue_conditions)}")
            if card.priority_note:
                text_parts.append(f"优先级 {card.priority}：{card.priority_note}")
            if card.counterexamples:
                text_parts.append(f"反例：{'；'.join(card.counterexamples)}")
            if card.case_pillars:
                text_parts.append(f"命例四柱：{' / '.join(card.case_pillars)}")
            if card.application_steps:
                text_parts.append(f"应用步骤：{'；'.join(card.application_steps)}")
            if card.disagreements:
                positions = "；".join(
                    f"{position.school}：{position.claim}" for position in card.disagreements
                )
                text_parts.append(f"流派分歧：{positions}")
            if card.prohibited_uses:
                text_parts.append(f"禁用范围：{'；'.join(card.prohibited_uses)}")
            rerank_parts = [card.title, card.rule, card.conclusion]
            rerank_parts.extend(card.conditions)
            rerank_parts.extend(card.exceptions)
            rerank_parts.extend(card.break_conditions)
            rerank_parts.extend(card.prohibited_uses)
            docs.append(
                RetrievalDocument(
                    id=card.id,
                    kind="knowledge_card",
                    layer=3,
                    title=card.title,
                    text="\n".join(text_parts),
                    source=self._format_refs(card.source_refs) or "项目种子知识卡",
                    school=card.school,
                    concepts=card.concepts,
                    conditions=card.conditions,
                    exclusions=card.exclusions,
                    trace_refs=[
                        card.id,
                        *card.annotation_refs,
                        *self._trace_ref_ids(card.source_refs),
                    ],
                    graph_refs=graph_refs,
                    retrieval_terms=graph_terms,
                    review_status=card.status,
                    verification_level=card.verification_level,
                    confidence=card.confidence,
                    warning=self._warning(card.status, card.unresolved_variants),
                    unresolved_variants=card.unresolved_variants,
                    content_sha256=card.content_sha256 or self._content_sha256(card.content),
                    version=card.version,
                    rerank_text="\n".join(part for part in rerank_parts if part),
                )
            )
        return docs

    def _graph_context(
        self,
        explicit_refs: Iterable[str],
        concepts: Iterable[str],
        scope: EvidenceScope,
    ) -> tuple[list[str], list[str]]:
        nodes = {item.id: item for item in self.graph_nodes}
        label_to_id = {
            label: node.id
            for node in self.graph_nodes
            if self._status_allowed(node.status, scope)
            for label in [node.name, *node.aliases]
        }
        allowed_nodes = {
            node.id for node in self.graph_nodes if self._status_allowed(node.status, scope)
        }
        refs = set(explicit_refs).intersection(allowed_nodes)
        refs.update(label_to_id[item] for item in concepts if item in label_to_id)
        adjacent: defaultdict[str, set[str]] = defaultdict(set)
        for edge in self.graph_edges:
            if not self._status_allowed(edge.status, scope):
                continue
            adjacent[edge.source].add(edge.target)
            adjacent[edge.target].add(edge.source)
        related = set(refs)
        for node_id in refs:
            related.update(adjacent[node_id])
        terms = sorted(
            {
                term
                for node_id in related
                if node_id in nodes
                for term in [nodes[node_id].name, *nodes[node_id].aliases]
            }
        )
        return sorted(refs), terms

    @staticmethod
    def _status_allowed(status: str, scope: EvidenceScope) -> bool:
        if scope == "reviewed_only":
            return status == "reviewed"
        return status in {"reviewed", "machine_verified"}

    @staticmethod
    def _content_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _warning(status: str, variants: list[str]) -> str:
        if status != "machine_verified":
            return ""
        warning = "机器校勘资料：已通过自动完整性检查，尚未经人工逐字校勘。"
        if variants:
            warning += f"仍有 {len(variants)} 项未解决异文。"
        return warning

    @staticmethod
    def _format_refs(refs: Iterable[SourceRef]) -> str:
        labels: list[str] = []
        for ref in refs:
            if ref.passage_id:
                labels.append(ref.passage_id)
            elif ref.annotation_id:
                labels.append(ref.annotation_id)
            else:
                label = " · ".join(part for part in [ref.source_id, ref.section] if part)
                labels.append(label + (f" · {ref.locator}" if ref.locator else ""))
        return "；".join(labels)

    @staticmethod
    def _direct_ref_ids(refs: Iterable[SourceRef]) -> list[str]:
        return [target for ref in refs for target in [ref.passage_id, ref.annotation_id] if target]

    def _trace_ref_ids(self, refs: Iterable[SourceRef]) -> list[str]:
        refs = list(refs)
        targets = self._direct_ref_ids(refs)
        legacy_targets = {
            (ref.source_id, ref.section)
            for ref in refs
            if ref.source_id and not ref.passage_id and not ref.annotation_id
        }
        for annotation in self.annotations:
            if any(
                (ref.source_id, ref.section) in legacy_targets for ref in annotation.source_refs
            ):
                targets.append(annotation.id)
        return list(dict.fromkeys(targets))
