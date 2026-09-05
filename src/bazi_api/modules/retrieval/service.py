from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import unicodedata
import uuid
from collections import Counter, defaultdict

import httpx

try:
    import bm25s
except ImportError:  # The transparent fallback is sufficient for the seed corpus.
    bm25s = None  # type: ignore[assignment]

from bazi_api.core.config import Settings
from bazi_api.core.errors import (
    InvalidUpstreamResponseError,
    ServiceUnavailableError,
    UpstreamServiceError,
)
from bazi_api.integrations.embeddings import EmbeddingProvider
from bazi_api.integrations.http import post_with_retries
from bazi_api.modules.charts.schemas import ChartFacts
from bazi_api.modules.knowledge.schemas import EvidenceScope, GraphEdge, GraphNode

from .cache import EmbeddingCache
from .schemas import RetrievalDocument, RetrievalHit

logger = logging.getLogger(__name__)

RETRIEVAL_TUNING_FIELDS = (
    "dense_recall_limit",
    "sparse_recall_limit",
    "rerank_limit",
    "bm25_k1",
    "bm25_b",
    "retrieval_rrf_k",
    "rerank_fused_weight",
    "rerank_model_weight",
    "graph_boost_per_match",
    "graph_boost_max",
    "concept_boost_per_match",
    "concept_boost_max",
    "evidence_chain_score_ratio",
    "bm25_heading_boost",
    "bm25_topic_boost",
    "exact_title_boost",
    "chapter_heading_boost",
    "chapter_topic_boost",
)

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
    words = re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]", normalize_retrieval_text(text))
    chinese = [token for token in words if len(token) == 1 and "\u3400" <= token <= "\u9fff"]
    compact = "".join(chinese)
    bigrams = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return words + bigrams


def query_mentions_term(query: str, term: str) -> bool:
    """Match concepts conservatively; a Han single character needs a real boundary."""

    normalized_query = normalize_retrieval_text(query)
    normalized_term = normalize_retrieval_text(term)
    if not normalized_term:
        return False
    if len(normalized_term) == 1 and "\u3400" <= normalized_term <= "\u9fff":
        pattern = rf"(?<![\u3400-\u9fff]){re.escape(normalized_term)}(?![\u3400-\u9fff])"
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
            left_boundary = not prefix or not ("\u3400" <= prefix[-1] <= "\u9fff")
            right_boundary = not suffix or not ("\u3400" <= suffix[0] <= "\u9fff")
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


class VectorIndex:
    def __init__(
        self,
        settings: Settings,
        documents: list[RetrievalDocument],
        vectors: list[list[float]],
        index_version: str,
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.vectors = {
            document.id: vector for document, vector in zip(documents, vectors, strict=True)
        }
        self.qdrant = None
        self.collection = f"bazi_knowledge_{index_version}"
        if settings.vector_backend == "qdrant" and vectors:
            self._initialize_qdrant(settings, documents, vectors)

    def _initialize_qdrant(
        self,
        settings: Settings,
        documents: list[RetrievalDocument],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client import QdrantClient, models

        embedded_qdrant = not settings.qdrant_url
        if settings.qdrant_url:
            self.qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=str(settings.qdrant_path))
        try:
            dimension = len(vectors[0])
            if self.qdrant.collection_exists(self.collection):
                info = self.qdrant.get_collection(self.collection)
                configured = info.config.params.vectors.size
                if configured != dimension:
                    self.qdrant.delete_collection(self.collection)
            if not self.qdrant.collection_exists(self.collection):
                self.qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, document.id)),
                    vector=vector,
                    payload={"document_id": document.id},
                )
                for document, vector in zip(documents, vectors, strict=True)
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points, wait=True)
            if embedded_qdrant:
                for collection in self.qdrant.get_collections().collections:
                    if (
                        collection.name.startswith("bazi_knowledge_")
                        and collection.name != self.collection
                    ):
                        self.qdrant.delete_collection(collection.name)
        except BaseException:
            self.qdrant.close()
            self.qdrant = None
            raise

    def search(self, vector: list[float], allowed: set[str], limit: int) -> list[tuple[str, float]]:
        if self.qdrant is not None:
            if not allowed:
                return []
            from qdrant_client import models

            response = self.qdrant.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchAny(any=sorted(allowed)),
                        )
                    ]
                ),
                limit=min(len(allowed), limit),
                with_payload=True,
            )
            hits: list[tuple[str, float]] = []
            for point in response.points:
                document_id = str(point.payload["document_id"])
                if document_id in allowed:
                    hits.append((document_id, float(point.score)))
                if len(hits) >= limit:
                    break
            return hits

        hits = []
        for document_id in allowed:
            candidate = self.vectors[document_id]
            score = float(sum(left * right for left, right in zip(vector, candidate, strict=True)))
            hits.append((document_id, score))
        return sorted(hits, key=lambda item: item[1], reverse=True)[:limit]

    def close(self) -> None:
        if self.qdrant is not None:
            self.qdrant.close()
            self.qdrant = None


class LexicalReranker:
    def __init__(self, rank_weight: float, lexical_weight: float) -> None:
        self.rank_weight = rank_weight
        self.lexical_weight = lexical_weight

    def score(self, query: str, documents: list[RetrievalDocument]) -> list[float]:
        query_tokens = set(tokenize_zh(query))
        scores: list[float] = []
        total = max(1, len(documents))
        for index, document in enumerate(documents):
            doc_tokens = set(tokenize_zh(f"{document.title} {document.text}"))
            overlap = len(query_tokens & doc_tokens)
            lexical = overlap / math.sqrt(max(1, len(query_tokens) * len(doc_tokens)))
            # The offline fallback should refine a strong fused ranking, not replace it.
            rank_prior = 1 - index / (total + 1)
            scores.append(rank_prior * self.rank_weight + lexical * self.lexical_weight)
        return scores


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, fallback: LexicalReranker, local_files_only: bool = False
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.model = None
        self.fallback = fallback
        self._lock = threading.Lock()

    def score(self, query: str, documents: list[RetrievalDocument]) -> list[float]:
        try:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import CrossEncoder

                    self.model = CrossEncoder(
                        self.model_name, local_files_only=self.local_files_only
                    )
                pairs = [
                    (
                        query,
                        (
                            f"{doc.title}\n{' '.join(doc.concepts)}\n"
                            f"{(doc.rerank_text or doc.text)[:900]}"
                        ),
                    )
                    for doc in documents
                ]
                return [
                    float(value) for value in self.model.predict(pairs, show_progress_bar=False)
                ]
        except Exception:  # Local model absence must not make the knowledge base unavailable.
            logger.warning(
                "reranker_fallback",
                extra={"provider": self.model_name},
                exc_info=True,
            )
            return self.fallback.score(query, documents)


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
    def __init__(
        self,
        documents: list[RetrievalDocument],
        embedding_provider: EmbeddingProvider,
        bm25: BM25Index,
        vectors: VectorIndex,
        reranker: LexicalReranker | CrossEncoderReranker,
        graph: KnowledgeGraphIndex,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.embedding_provider = embedding_provider
        self.bm25 = bm25
        self.vectors = vectors
        self.reranker = reranker
        self.graph = graph
        self.settings = settings
        self.http_client = http_client
        self.model_version = embedding_provider.model_version
        self.index_version = ""
        self.cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
        self._lightrag_aliases = self._build_lightrag_aliases(documents)

    @classmethod
    async def create(
        cls,
        settings: Settings,
        documents: list[RetrievalDocument],
        embedding_provider: EmbeddingProvider,
        graph_nodes: list[GraphNode] | None = None,
        graph_edges: list[GraphEdge] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> RetrievalService:
        texts = [cls._embedding_text(doc) for doc in documents]
        content_hashes = [
            hashlib.sha256(f"{doc.content_sha256}\n{text}".encode()).hexdigest()
            for doc, text in zip(documents, texts, strict=True)
        ]
        cache = await asyncio.to_thread(EmbeddingCache, settings.embedding_cache_path)
        try:
            try:
                embeddings, cache_stats = await cache.vectors(
                    embedding_provider, content_hashes, texts
                )
            except Exception:
                logger.warning(
                    "embedding_provider_fallback",
                    extra={"provider": embedding_provider.model_version},
                    exc_info=True,
                )
                from bazi_api.integrations.embeddings import HashEmbeddingProvider

                embedding_provider = HashEmbeddingProvider()
                embeddings, cache_stats = await cache.vectors(
                    embedding_provider, content_hashes, texts
                )
        finally:
            await asyncio.to_thread(cache.close)
        lexical_reranker = LexicalReranker(
            rank_weight=settings.rerank_fused_weight,
            lexical_weight=settings.rerank_model_weight,
        )
        reranker: LexicalReranker | CrossEncoderReranker
        if settings.reranker_provider == "cross_encoder":
            reranker = CrossEncoderReranker(
                settings.reranker_model,
                lexical_reranker,
                settings.local_models_only,
            )
        else:
            reranker = lexical_reranker
        tuning_payload = json.dumps(
            retrieval_tuning(settings), sort_keys=True, separators=(",", ":")
        )
        vector_payload = "\n".join([embedding_provider.model_version, *sorted(content_hashes)])
        vector_index_version = hashlib.sha256(vector_payload.encode("utf-8")).hexdigest()[:16]
        index_payload = "\n".join(
            [embedding_provider.model_version, *sorted(content_hashes), tuning_payload]
        )
        index_version = hashlib.sha256(index_payload.encode("utf-8")).hexdigest()[:16]
        bm25 = await asyncio.to_thread(
            BM25Index,
            documents,
            settings.bm25_heading_boost,
            settings.bm25_topic_boost,
            settings.bm25_k1,
            settings.bm25_b,
        )
        vectors = await asyncio.to_thread(
            VectorIndex, settings, documents, embeddings, vector_index_version
        )
        service = cls(
            documents=documents,
            embedding_provider=embedding_provider,
            bm25=bm25,
            vectors=vectors,
            reranker=reranker,
            graph=KnowledgeGraphIndex(graph_nodes or [], graph_edges or []),
            settings=settings,
            http_client=http_client,
        )
        service.cache_stats = cache_stats
        service.model_version = embedding_provider.model_version
        service.index_version = index_version
        return service

    def close(self) -> None:
        self.vectors.close()

    @staticmethod
    def _embedding_text(document: RetrievalDocument) -> str:
        body = document.normalized_text or document.text
        return normalize_retrieval_text(
            f"{document.title}\n{body}\n{' '.join(document.concepts)}\n"
            f"{' '.join(document.retrieval_terms)}"
        )

    async def search(
        self,
        query: str,
        chart: ChartFacts,
        school: str,
        mode: str,
        limit: int = 6,
        evidence_scope: EvidenceScope = "reviewed_only",
        allowed_schools: list[str] | None = None,
        preferred_schools: list[str] | None = None,
    ) -> list[RetrievalHit]:
        schools = set(allowed_schools or [school])
        allowed = {
            document.id
            for document in self.documents.values()
            if self._eligible(document, chart, schools, evidence_scope)
        }
        if mode == "lightrag":
            hits = await self._search_lightrag(query, allowed, limit)
            self._apply_school_priority(preferred_schools or [], hits)
            return hits
        chart_terms = self._chart_query_terms(query, chart)
        graph_query = " ".join([query, *chart_terms])
        expanded_terms, related_nodes = self.graph.expand(graph_query, evidence_scope)
        retrieval_query = normalize_retrieval_text(" ".join([query, *chart_terms, *expanded_terms]))
        if mode == "dense":
            dense = await self._dense_search(retrieval_query, allowed)
            if dense:
                hits = self._hits_from_single(dense, "dense", limit)
            else:
                sparse_fallback = await asyncio.to_thread(
                    self.bm25.search, retrieval_query, allowed, self.settings.sparse_recall_limit
                )
                hits = self._hits_from_single(sparse_fallback, "bm25-fallback", limit)
            hits = self._diversify_hits(hits, limit)
            self._apply_school_priority(preferred_schools or [], hits)
            return self._expand_evidence_chain(hits, allowed, limit)
        dense, sparse = await asyncio.gather(
            self._dense_search(retrieval_query, allowed),
            asyncio.to_thread(
                self.bm25.search,
                retrieval_query,
                allowed,
                self.settings.sparse_recall_limit,
            ),
        )
        fused = self._rrf({"dense": dense, "bm25": sparse}, self.settings.retrieval_rrf_k)
        hits = [
            RetrievalHit(
                document=self.documents[document_id],
                score=score,
                matched_by=sorted(components),
                component_scores=raw,
            )
            for document_id, score, components, raw in fused
        ]
        self._apply_title_boost(query, hits)
        self._apply_concept_boost(query, hits)
        self._apply_school_priority(preferred_schools or [], hits)
        self._apply_graph_boost(related_nodes, hits)
        self._apply_chart_context_boost(chart_terms, chart, hits)
        if mode == "hybrid_rerank" and hits:
            candidates = hits[: self.settings.rerank_limit]
            remainder = hits[self.settings.rerank_limit :]
            rerank_scores = await asyncio.to_thread(
                self.reranker.score, query, [hit.document for hit in candidates]
            )
            for hit, rerank_score in zip(candidates, rerank_scores, strict=True):
                hit.component_scores["reranker"] = rerank_score
                hit.score = (
                    hit.score * self.settings.rerank_fused_weight
                    + rerank_score * self.settings.rerank_model_weight
                )
                hit.matched_by.append("reranker")
            candidates.sort(key=lambda hit: hit.score, reverse=True)
            hits = [*candidates, *remainder]
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
    def _chart_query_terms(query: str, chart: ChartFacts) -> list[str]:
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
        chart_terms: list[str], chart: ChartFacts, hits: list[RetrievalHit]
    ) -> None:
        if not chart_terms:
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

    async def _dense_search(self, query: str, allowed: set[str]) -> list[tuple[str, float]]:
        try:
            query_vector = (await self.embedding_provider.embed([query]))[0]
            return await asyncio.to_thread(
                self.vectors.search,
                query_vector,
                allowed,
                self.settings.dense_recall_limit,
            )
        except Exception:
            logger.warning(
                "dense_retrieval_fallback",
                extra={"provider": self.embedding_provider.model_version},
                exc_info=True,
            )
            return []

    async def _search_lightrag(
        self, query: str, allowed: set[str], limit: int
    ) -> list[RetrievalHit]:
        if not self.settings.lightrag_base_url:
            raise ServiceUnavailableError("LightRAG 尚未配置")
        headers = {"Content-Type": "application/json"}
        if self.settings.lightrag_api_key:
            headers["X-API-Key"] = self.settings.lightrag_api_key
        url = f"{self.settings.lightrag_base_url.rstrip('/')}/query/data"
        request_payload: dict[str, object] = {
            "query": query,
            "mode": "mix",
            "chunk_top_k": max(limit, 5),
            "enable_rerank": True,
        }
        try:
            if self.http_client is not None:
                response = await post_with_retries(
                    self.http_client,
                    url,
                    headers=headers,
                    payload=request_payload,
                    timeout=120,
                    max_retries=self.settings.http_request_retries,
                    operation="lightrag",
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await post_with_retries(
                        client,
                        url,
                        headers=headers,
                        payload=request_payload,
                        timeout=120,
                        max_retries=self.settings.http_request_retries,
                        operation="lightrag",
                    )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("lightrag_request_failed", exc_info=True)
            raise UpstreamServiceError("LightRAG 查询服务暂时不可用") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidUpstreamResponseError("LightRAG 返回了无效 JSON") from exc
        if not isinstance(payload, dict):
            raise InvalidUpstreamResponseError("LightRAG 响应顶层必须是对象")
        if payload.get("status") != "success":
            raise InvalidUpstreamResponseError("LightRAG 返回了失败状态")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise InvalidUpstreamResponseError("LightRAG 响应缺少 data 对象")
        chunks = data.get("chunks", [])
        entities = data.get("entities", [])
        if not isinstance(chunks, list) or not isinstance(entities, list):
            raise InvalidUpstreamResponseError("LightRAG 候选列表格式无效")
        candidates: list[tuple[str, str]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise InvalidUpstreamResponseError("LightRAG chunk 格式无效")
            item_id = chunk.get("chunk_id", f"chunk-{len(candidates)}")
            source = chunk.get("file_path", "")
            if not isinstance(item_id, str) or not isinstance(source, str):
                raise InvalidUpstreamResponseError("LightRAG chunk 标识格式无效")
            candidates.append((item_id, source))
        if not candidates:
            for entity in entities:
                if not isinstance(entity, dict):
                    raise InvalidUpstreamResponseError("LightRAG entity 格式无效")
                item_id = entity.get("entity_name", f"entity-{len(candidates)}")
                source = entity.get("file_path", "")
                if not isinstance(item_id, str) or not isinstance(source, str):
                    raise InvalidUpstreamResponseError("LightRAG entity 标识格式无效")
                candidates.append((item_id, source))
        hits: list[RetrievalHit] = []
        selected: set[str] = set()
        rejected = 0
        for rank, (item_id, source) in enumerate(candidates, start=1):
            document = self._resolve_lightrag_document(item_id, source)
            if document is None or document.id not in allowed or document.id in selected:
                rejected += 1
                continue
            selected.add(document.id)
            hits.append(
                RetrievalHit(
                    document=document,
                    score=1 / rank,
                    matched_by=["lightrag-mix"],
                    component_scores={"lightrag": 1 / rank},
                )
            )
            if len(hits) >= limit:
                break
        if rejected:
            logger.warning(
                "lightrag_untrusted_candidates_rejected",
                extra={"provider": "lightrag", "hits": rejected},
            )
        return hits

    @staticmethod
    def _build_lightrag_aliases(
        documents: list[RetrievalDocument],
    ) -> dict[str, str | None]:
        aliases: dict[str, str | None] = {}
        for document in documents:
            safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", document.id).strip("-")
            for alias in {document.id, safe_id}:
                if alias in aliases and aliases[alias] != document.id:
                    aliases[alias] = None
                else:
                    aliases[alias] = document.id
        return aliases

    def _resolve_lightrag_document(self, item_id: str, source: str) -> RetrievalDocument | None:
        filename = source.replace("\\", "/").rsplit("/", 1)[-1]
        source_alias = filename[:-3] if filename.lower().endswith(".md") else filename
        for alias in (item_id, source_alias):
            document_id = self._lightrag_aliases.get(alias)
            if document_id is not None:
                return self.documents[document_id]
        return None

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
        base = hits[:limit]
        parent: RetrievalHit | None = None
        canonical: RetrievalDocument | None = None
        for candidate in base:
            if candidate.document.kind not in {"knowledge_card", "modern_annotation"}:
                continue
            resolved = self._find_canonical_reference(candidate.document, allowed)
            if resolved is not None:
                parent = candidate
                canonical = resolved
                break
        if parent is None or canonical is None:
            return base
        linked = next(
            (hit for hit in base if hit.document.id == canonical.id),
            RetrievalHit(
                document=canonical,
                score=parent.score * self.settings.evidence_chain_score_ratio,
                matched_by=["evidence_chain"],
                component_scores={"evidence_chain": parent.score},
            ),
        )
        without_linked = [hit for hit in base if hit.document.id != canonical.id]
        parent_index = without_linked.index(parent)
        ordered = [
            *without_linked[: parent_index + 1],
            linked,
            *without_linked[parent_index + 1 :],
        ]
        return ordered[:limit]

    def _find_canonical_reference(
        self, document: RetrievalDocument, allowed: set[str]
    ) -> RetrievalDocument | None:
        queue = list(document.trace_refs)
        visited: set[str] = set()
        while queue:
            reference = queue.pop(0)
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
        self, ranking: list[tuple[str, float]], component: str, limit: int
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                document=self.documents[document_id],
                score=score,
                matched_by=[component],
                component_scores={component: score},
            )
            for document_id, score in ranking[:limit]
        ]

    def _rrf(
        self, rankings: dict[str, list[tuple[str, float]]], k: int
    ) -> list[tuple[str, float, list[str], dict[str, float]]]:
        scores: defaultdict[str, float] = defaultdict(float)
        component_scores: defaultdict[str, list[str]] = defaultdict(list)
        raw_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
        for component, ranking in rankings.items():
            for rank, (document_id, raw_score) in enumerate(ranking, start=1):
                scores[document_id] += 1 / (k + rank)
                component_scores[document_id].append(component)
                raw_scores[document_id][component] = raw_score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            (document_id, score, component_scores[document_id], raw_scores[document_id])
            for document_id, score in ranked
        ]

    @staticmethod
    def _eligible(
        document: RetrievalDocument,
        chart: ChartFacts,
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
        selected: list[RetrievalHit] = []
        seen_families: set[str] = set()
        chapter_counts: Counter[str] = Counter()
        for hit in hits:
            canonical_refs = [
                ref for ref in hit.document.trace_refs if ref != hit.document.id and "-ch" in ref
            ]
            family = canonical_refs[0] if canonical_refs else hit.document.id
            chapter_match = re.search(r"-ch(\d+)-", family)
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
