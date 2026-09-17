from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque

try:
    import bm25s
except ImportError:  # The transparent fallback is sufficient for the seed corpus.
    bm25s = None  # type: ignore[assignment]

from bazi_api.core.config import Settings
from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.knowledge.schemas import EvidenceScope, GraphEdge, GraphNode

from .schemas import RetrievalDocument, RetrievalHit

logger = logging.getLogger(__name__)

RETRIEVAL_TUNING_FIELDS = (
    "sparse_recall_limit",
    "bm25_k1",
    "bm25_b",
    "bm25_heading_boost",
    "bm25_topic_boost",
    "exact_title_boost",
    "chapter_heading_boost",
    "chapter_topic_boost",
    "graph_boost_per_match",
    "graph_boost_max",
    "concept_boost_per_match",
    "concept_boost_max",
    "evidence_chain_score_ratio",
)

MODEL_VERSION = "bm25-lexicon-v1"

TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "陰": "阴",
        "陽": "阳",
        "財": "财",
        "剋": "克",
        "殺": "杀",
        "傷": "伤",
        "運": "运",
        "會": "会",
        "衝": "冲",
        "歲": "岁",
        "時": "时",
        "論": "论",
        "與": "与",
        "為": "为",
        "從": "从",
        "詮": "诠",
        "書": "书",
        "體": "体",
        "氣": "气",
        "祿": "禄",
        "應": "应",
        "變": "变",
        "敗": "败",
        "雜": "杂",
        "格": "格",
    }
)


def normalize_retrieval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(TRADITIONAL_TO_SIMPLIFIED)
    normalized = normalized.lower().replace("提纲", "月令 提纲")
    normalized = normalized.replace("煞", "杀")
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize_zh(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+|[㐀-鿿]", normalize_retrieval_text(text))
    chinese = [token for token in words if len(token) == 1 and "㐀" <= token <= "鿿"]
    compact = "".join(chinese)
    bigrams = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return words + bigrams


def query_mentions_term(query: str, term: str) -> bool:
    """Match concepts conservatively; a Han single character needs a real boundary."""

    normalized_query = normalize_retrieval_text(query)
    normalized_term = normalize_retrieval_text(term)
    if not normalized_term:
        return False
    if len(normalized_term) == 1 and "㐀" <= normalized_term <= "鿿":
        pattern = rf"(?<![㐀-鿿]){re.escape(normalized_term)}(?![㐀-鿿])"
        if re.search(pattern, normalized_query) is not None:
            return True
        left_cues = ("的", "属", "为", "是", "看", "论", "中", "里", "与", "和")
        right_cues = (
            "的",
            "是",
            "属",
            "为",
            "多",
            "少",
            "旺",
            "衰",
            "强",
            "弱",
            "与",
            "和",
            "能",
            "会",
            "可",
            "不",
            "应该",
            "如何",
            "怎么",
            "代表",
            "吗",
            "呢",
        )
        for match in re.finditer(re.escape(normalized_term), normalized_query):
            prefix = normalized_query[: match.start()]
            suffix = normalized_query[match.end() :]
            left_boundary = not prefix or not ("㐀" <= prefix[-1] <= "鿿")
            right_boundary = not suffix or not ("㐀" <= suffix[0] <= "鿿")
            left_ok = left_boundary or prefix.endswith(left_cues)
            right_ok = right_boundary or suffix.startswith(right_cues)
            if left_ok and right_ok:
                return True
        return False
    return normalized_term in normalized_query


def chapter_heading_terms(heading: str) -> list[str]:
    topic = heading.removeprefix("论").strip()
    simplified = re.sub(r"[一二三四五六七八九十类]$", "", topic).strip()
    base = re.sub(r"(?:说|法|论|歌|诀|类)$", "", simplified).strip()
    return list(
        dict.fromkeys(item for item in (heading, topic, simplified, base) if len(item) >= 2)
    )


def primary_query_text(query: str) -> str:
    return query.split("\n检索扩展：", 1)[0]


def retrieval_tuning(settings: Settings) -> dict[str, int | float]:
    return {name: getattr(settings, name) for name in RETRIEVAL_TUNING_FIELDS}


class BM25Index:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        heading_boost: float,
        topic_boost: float,
        k1: float,
        b: float,
    ) -> None:
        self.documents = documents
        self.heading_boost = heading_boost
        self.topic_boost = topic_boost
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize_zh(self._search_text(doc)) for doc in documents]
        self.index = None
        self.tokenizer = None
        if bm25s is not None:
            self.tokenizer = bm25s.tokenization.Tokenizer(
                stemmer=None,
                stopwords=[],
                splitter=lambda text: text.split(),
            )
            tokens = self.tokenizer.tokenize(
                [" ".join(tokens) for tokens in self.corpus_tokens], show_progress=False
            )
            self.index = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
            self.index.index(tokens, show_progress=False)
        self.average_length = sum(map(len, self.corpus_tokens)) / max(1, len(self.corpus_tokens))
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            self.document_frequency.update(set(tokens))

    @staticmethod
    def _search_text(document: RetrievalDocument) -> str:
        return " ".join(
            [
                document.title,
                document.normalized_text or document.text,
                *document.concepts,
                *document.conditions,
                *document.retrieval_terms,
            ]
        )

    def search(self, query: str, allowed: set[str], limit: int) -> list[tuple[str, float]]:
        if self.index is not None and self.tokenizer is not None:
            query_tokens = self.tokenizer.tokenize(
                [" ".join(tokenize_zh(query))], show_progress=False
            )
            # bm25s cannot filter during retrieval. Fetch the full ranking before
            # applying school/status eligibility so a large multi-school corpus
            # cannot discard an exact-title match prematurely.
            count = len(self.documents)
            indices, scores = self.index.retrieve(query_tokens, k=count, show_progress=False)
            hits: list[tuple[str, float]] = []
            for index, score in zip(indices[0].tolist(), scores[0].tolist(), strict=True):
                document = self.documents[int(index)]
                if document.id in allowed:
                    hits.append(
                        (
                            document.id,
                            float(score) + self._title_match_bonus(query, document),
                        )
                    )
            return sorted(hits, key=lambda item: item[1], reverse=True)[:limit]
        return self._transparent_search(query, allowed, limit)

    def _title_match_bonus(self, query: str, document: RetrievalDocument) -> float:
        match = re.search(r"第\d+章\s+([^/]+)$", document.title)
        if match is None:
            return 0.0
        normalized_query = normalize_retrieval_text(primary_query_text(query))
        heading = normalize_retrieval_text(match.group(1)).strip()
        if heading and heading in normalized_query:
            return self.heading_boost
        if any(term in normalized_query for term in chapter_heading_terms(heading)[1:]):
            return self.topic_boost
        return 0.0

    def _transparent_search(
        self, query: str, allowed: set[str], limit: int
    ) -> list[tuple[str, float]]:
        query_tokens = set(tokenize_zh(query))
        total = len(self.documents)
        ranked: list[tuple[str, float]] = []
        for document, tokens in zip(self.documents, self.corpus_tokens, strict=True):
            if document.id not in allowed:
                continue
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                frequency_docs = self.document_frequency[token]
                inverse_frequency = math.log(
                    1 + (total - frequency_docs + 0.5) / (frequency_docs + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(tokens) / max(1, self.average_length)
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            score += self._title_match_bonus(query, document)
            ranked.append((document.id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]


class KnowledgeGraphIndex:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        self.nodes = {node.id: node for node in nodes}
        self.edges = edges

    def expand(self, query: str, scope: EvidenceScope) -> tuple[list[str], set[str]]:
        allowed_statuses = (
            {"reviewed"} if scope == "reviewed_only" else {"reviewed", "machine_verified"}
        )
        allowed_nodes = {node.id for node in self.nodes.values() if node.status in allowed_statuses}
        direct = {
            node.id
            for node in self.nodes.values()
            if node.id in allowed_nodes
            and any(query_mentions_term(query, label) for label in [node.name, *node.aliases])
        }
        adjacent: defaultdict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            if edge.status not in allowed_statuses:
                continue
            if edge.source not in allowed_nodes or edge.target not in allowed_nodes:
                continue
            adjacent[edge.source].add(edge.target)
            adjacent[edge.target].add(edge.source)
        related = set(direct)
        for node_id in direct:
            related.update(adjacent[node_id])
        terms = sorted(
            {
                label
                for node_id in related
                if self.nodes[node_id].type != "knowledge_card"
                for label in [self.nodes[node_id].name, *self.nodes[node_id].aliases]
                if label and not query_mentions_term(query, label)
            }
        )
        return terms, related


class RetrievalService:
    """按需查典核验通道：单一的 bm25 词法检索，无向量索引与重排。"""

    def __init__(
        self,
        documents: list[RetrievalDocument],
        bm25: BM25Index,
        graph: KnowledgeGraphIndex,
        settings: Settings,
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.bm25 = bm25
        self.graph = graph
        self.settings = settings
        self.model_version = MODEL_VERSION
        self.index_version = ""

    @classmethod
    async def create(
        cls,
        settings: Settings,
        documents: list[RetrievalDocument],
        graph_nodes: list[GraphNode] | None = None,
        graph_edges: list[GraphEdge] | None = None,
    ) -> RetrievalService:
        tuning_payload = json.dumps(
            retrieval_tuning(settings), sort_keys=True, separators=(",", ":")
        )
        content_hashes = sorted(document.content_sha256 for document in documents)
        index_version = hashlib.sha256(
            "\n".join([MODEL_VERSION, *content_hashes, tuning_payload]).encode("utf-8")
        ).hexdigest()[:16]
        bm25 = await asyncio.to_thread(
            BM25Index,
            documents,
            settings.bm25_heading_boost,
            settings.bm25_topic_boost,
            settings.bm25_k1,
            settings.bm25_b,
        )
        service = cls(
            documents=documents,
            bm25=bm25,
            graph=KnowledgeGraphIndex(graph_nodes or [], graph_edges or []),
            settings=settings,
        )
        service.index_version = index_version
        return service

    async def search(
        self,
        query: str,
        chart: ChartFacts | None,
        school: str,
        mode: str,
        limit: int = 6,
        evidence_scope: EvidenceScope = "reviewed_only",
        allowed_schools: list[str] | None = None,
        preferred_schools: list[str] | None = None,
    ) -> list[RetrievalHit]:
        if mode != "bm25":
            raise ValueError(f"unknown mode: {mode}")
        schools = set(allowed_schools or [school])
        allowed = {
            document.id
            for document in self.documents.values()
            if self._eligible(document, chart, schools, evidence_scope)
        }
        chart_terms = self._chart_query_terms(query, chart)
        graph_query = " ".join([query, *chart_terms])
        expanded_terms, related_nodes = self.graph.expand(graph_query, evidence_scope)
        retrieval_query = normalize_retrieval_text(
            " ".join([query, *chart_terms, *expanded_terms])
        )
        sparse = await asyncio.to_thread(
            self.bm25.search, retrieval_query, allowed, self.settings.sparse_recall_limit
        )
        hits = self._hits_from_single(sparse, "bm25")
        self._apply_title_boost(query, hits)
        self._apply_concept_boost(query, hits)
        self._apply_school_priority(preferred_schools or [], hits)
        self._apply_graph_boost(related_nodes, hits)
        self._apply_chart_context_boost(chart_terms, chart, hits)
        hits = self._diversify_hits(hits, limit)
        return self._expand_evidence_chain(hits, allowed, limit)

    @staticmethod
    def _apply_school_priority(
        preferred_schools: list[str], hits: list[RetrievalHit]
    ) -> None:
        if not preferred_schools:
            return
        count = len(preferred_schools)
        for hit in hits:
            if hit.document.school not in preferred_schools:
                continue
            rank = preferred_schools.index(hit.document.school)
            bonus = 0.01 * (count - rank)
            hit.score += bonus
            hit.component_scores["expert_school_priority"] = bonus
            hit.matched_by.append("expert_school_priority")
        hits.sort(key=lambda hit: hit.score, reverse=True)

    @staticmethod
    def _chart_query_terms(query: str, chart: ChartFacts | None) -> list[str]:
        if chart is None:
            return []
        applied_markers = (
            "这个命格",
            "我的命格",
            "命格如何",
            "这个格局",
            "我的格局",
            "格局如何",
            "这个八字",
            "我的八字",
            "这个命局",
            "我的命局",
            "命局如何",
            "这个命盘",
            "我的命盘",
            "这盘",
            "此盘",
            "本盘",
            "帮我看",
            "分析命盘",
        )
        if not any(marker in query for marker in applied_markers):
            return []
        terms = [
            "月令",
            chart.month_command.branch,
            chart.month_command.main_hidden_stem,
            chart.month_command.ten_god,
            *(candidate.name for candidate in chart.pattern_candidates),
            *(relation.label for relation in chart.structural_relations),
        ]
        return list(dict.fromkeys(term for term in terms if term))

    @staticmethod
    def _apply_chart_context_boost(
        chart_terms: list[str], chart: ChartFacts | None, hits: list[RetrievalHit]
    ) -> None:
        if not chart_terms or chart is None:
            return
        candidate_terms = {
            term
            for candidate in chart.pattern_candidates
            for term in (candidate.name, candidate.ten_god)
        }
        for hit in hits:
            matched = [term for term in candidate_terms if term in hit.document.title]
            if not matched:
                continue
            bonus = 0.2
            hit.score += bonus
            hit.component_scores["chart_context"] = bonus
            hit.matched_by.append("chart_context")
        hits.sort(key=lambda hit: hit.score, reverse=True)

    def _apply_graph_boost(self, related_nodes: set[str], hits: list[RetrievalHit]) -> None:
        if not related_nodes:
            return
        for hit in hits:
            matched = related_nodes.intersection(hit.document.graph_refs)
            if not matched:
                continue
            bonus = min(
                self.settings.graph_boost_max,
                self.settings.graph_boost_per_match * len(matched),
            )
            hit.score += bonus
            hit.component_scores["knowledge_graph"] = bonus
            hit.matched_by.append("knowledge_graph")
        hits.sort(key=lambda hit: hit.score, reverse=True)

    def _expand_evidence_chain(
        self, hits: list[RetrievalHit], allowed: set[str], limit: int
    ) -> list[RetrievalHit]:
        base = hits[: max(0, limit)]
        by_id = {hit.document.id: hit for hit in base}
        parent: RetrievalHit | None = None
        canonical: RetrievalDocument | None = None
        for candidate in base:
            if candidate.document.kind not in {"knowledge_card", "modern_annotation"}:
                continue
            resolved = self._find_canonical_reference(candidate.document, allowed)
            if resolved is not None:
                parent, canonical = candidate, resolved
                break
        if parent is None or canonical is None:
            return base

        linked = by_id.get(canonical.id)
        if linked is None:
            linked = RetrievalHit(
                document=canonical,
                score=parent.score * self.settings.evidence_chain_score_ratio,
                matched_by=["evidence_chain"],
                component_scores={"evidence_chain": parent.score},
            )
        without_linked = [hit for hit in base if hit.document.id != canonical.id]
        parent_index = without_linked.index(parent)
        ordered = [
            *without_linked[: parent_index + 1],
            linked,
            *without_linked[parent_index + 1 :],
        ][:limit]
        seen = {hit.document.id for hit in ordered}
        if limit < 4 or canonical.id not in seen or len(ordered) >= limit:
            return ordered

        # Additional provenance may fill spare slots, never replace ranked evidence.
        for candidate in ordered:
            if candidate.document.id == parent.document.id or candidate.document.kind not in {
                "knowledge_card",
                "modern_annotation",
            }:
                continue
            canonical = self._find_canonical_reference(candidate.document, allowed)
            if canonical is None or canonical.id in seen:
                continue
            linked = by_id.get(canonical.id)
            if linked is None:
                linked = RetrievalHit(
                    document=canonical,
                    score=candidate.score * self.settings.evidence_chain_score_ratio,
                    matched_by=["evidence_chain"],
                    component_scores={"evidence_chain": candidate.score},
                )
            return [*ordered, linked]
        return ordered

    def _find_canonical_reference(
        self, document: RetrievalDocument, allowed: set[str]
    ) -> RetrievalDocument | None:
        queue = deque(document.trace_refs)
        visited: set[str] = set()
        while queue:
            reference = queue.popleft()
            if reference in visited or reference not in allowed:
                continue
            visited.add(reference)
            candidate = self.documents.get(reference)
            if candidate is None:
                continue
            if candidate.kind == "canonical_passage":
                return candidate
            queue.extend(candidate.trace_refs)
        return None

    def _apply_title_boost(self, query: str, hits: list[RetrievalHit]) -> None:
        normalized_query = normalize_retrieval_text(primary_query_text(query))
        for hit in hits:
            document_title = normalize_retrieval_text(hit.document.title).strip()
            work_title, separator, _ = document_title.partition(" · ")
            if separator and len(work_title) >= 2 and work_title in normalized_query:
                bonus = self.settings.exact_title_boost
                hit.score += bonus
                hit.component_scores["work_title"] = bonus
                hit.matched_by.append("work_title")
            if len(document_title) >= 2 and document_title in normalized_query:
                bonus = self.settings.exact_title_boost
                hit.score += bonus
                hit.component_scores["exact_title"] = bonus
                hit.matched_by.append("exact_title")
            match = re.search(r"第\d+章\s+([^/]+)$", hit.document.title)
            if match is None:
                continue
            heading = normalize_retrieval_text(match.group(1)).strip()
            if heading and heading in normalized_query:
                bonus = self.settings.chapter_heading_boost
            elif any(term in normalized_query for term in chapter_heading_terms(heading)[1:]):
                bonus = self.settings.chapter_topic_boost
            else:
                continue
            hit.score += bonus
            hit.component_scores["chapter_heading"] = bonus
            hit.matched_by.append("chapter_heading")
        hits.sort(key=lambda hit: hit.score, reverse=True)

    def _apply_concept_boost(self, query: str, hits: list[RetrievalHit]) -> None:
        for hit in hits:
            matched = [
                concept for concept in hit.document.concepts if query_mentions_term(query, concept)
            ]
            if not matched:
                continue
            bonus = min(
                self.settings.concept_boost_max,
                self.settings.concept_boost_per_match * len(matched),
            )
            hit.score += bonus
            hit.component_scores["concept"] = bonus
            hit.matched_by.append("concept")
        hits.sort(key=lambda hit: hit.score, reverse=True)

    def _hits_from_single(
        self, ranking: list[tuple[str, float]], component: str
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                document=self.documents[document_id],
                score=score,
                matched_by=[component],
                component_scores={component: score},
            )
            for document_id, score in ranking
        ]

    @staticmethod
    def _eligible(
        document: RetrievalDocument,
        chart: ChartFacts | None,
        schools: set[str],
        evidence_scope: EvidenceScope,
    ) -> bool:
        allowed_statuses = (
            {"reviewed"} if evidence_scope == "reviewed_only" else {"reviewed", "machine_verified"}
        )
        if document.review_status not in allowed_statuses:
            return False
        if document.school not in {*schools, "基础共识"}:
            return False
        if chart is not None:
            facts = {
                "day_master": chart.day_master,
                "day_master_element": chart.day_master_element,
                "day_master_yin_yang": chart.day_master_yin_yang,
            }
            for condition in document.conditions:
                if "=" not in condition:
                    continue
                key, expected = (part.strip() for part in condition.split("=", 1))
                if key in facts and facts[key] != expected:
                    return False
            for exclusion in document.exclusions:
                if "=" not in exclusion:
                    continue
                key, expected = (part.strip() for part in exclusion.split("=", 1))
                if key in facts and facts[key] == expected:
                    return False
        return True

    @staticmethod
    def _diversify_hits(hits: list[RetrievalHit], limit: int) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        selected: list[RetrievalHit] = []
        seen_families: set[str] = set()
        chapter_counts: Counter[str] = Counter()
        for hit in hits:
            canonical_refs = [
                ref for ref in hit.document.trace_refs if ref != hit.document.id and "-ch" in ref
            ]
            family = canonical_refs[0] if canonical_refs else hit.document.id
            chapter_match = re.search(r"^(.+?-ch\d+)-", family)
            chapter = chapter_match.group(1) if chapter_match else ""
            if family in seen_families or (chapter and chapter_counts[chapter] >= 2):
                continue
            selected.append(hit)
            seen_families.add(family)
            if chapter:
                chapter_counts[chapter] += 1
            if len(selected) >= limit:
                break
        return selected
