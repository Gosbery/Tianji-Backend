from __future__ import annotations

import math
import re
import uuid
from collections import Counter, defaultdict

import httpx

try:
    import bm25s
except ImportError:  # The transparent fallback is sufficient for the seed corpus.
    bm25s = None  # type: ignore[assignment]

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import EmbeddingProvider
from bazi_api.modules.charts.schemas import ChartFacts

from .schemas import RetrievalDocument, RetrievalHit


def tokenize_zh(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]", text.lower())
    chinese = [token for token in words if len(token) == 1 and "\u3400" <= token <= "\u9fff"]
    compact = "".join(chinese)
    bigrams = [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    return words + bigrams


class BM25Index:
    def __init__(self, documents: list[RetrievalDocument]) -> None:
        self.documents = documents
        self.corpus_tokens = [tokenize_zh(self._search_text(doc)) for doc in documents]
        self.index = None
        self.tokenizer = None
        if bm25s is not None:
            self.tokenizer = bm25s.tokenization.Tokenizer(
                stemmer=None,
                stopwords=[],
                splitter=lambda text: text.split(),
            )
            tokens = self.tokenizer.tokenize([" ".join(tokens) for tokens in self.corpus_tokens])
            self.index = bm25s.BM25(method="lucene")
            self.index.index(tokens, show_progress=False)
        self.average_length = sum(map(len, self.corpus_tokens)) / max(1, len(self.corpus_tokens))
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.corpus_tokens:
            self.document_frequency.update(set(tokens))

    @staticmethod
    def _search_text(document: RetrievalDocument) -> str:
        return " ".join([document.title, document.text, *document.concepts, *document.conditions])

    def search(self, query: str, allowed: set[str], limit: int) -> list[tuple[str, float]]:
        if self.index is not None and self.tokenizer is not None:
            query_tokens = self.tokenizer.tokenize([" ".join(tokenize_zh(query))])
            count = min(len(self.documents), max(limit * 4, limit))
            indices, scores = self.index.retrieve(query_tokens, k=count, show_progress=False)
            hits: list[tuple[str, float]] = []
            for index, score in zip(indices[0].tolist(), scores[0].tolist(), strict=True):
                document = self.documents[int(index)]
                if document.id in allowed:
                    hits.append((document.id, float(score)))
                if len(hits) >= limit:
                    break
            return hits
        return self._transparent_search(query, allowed, limit)

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
                denominator = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * len(tokens) / max(1, self.average_length)
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            ranked.append((document.id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]


class VectorIndex:
    def __init__(
        self,
        settings: Settings,
        documents: list[RetrievalDocument],
        vectors: list[list[float]],
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.vectors = {
            document.id: vector for document, vector in zip(documents, vectors, strict=True)
        }
        self.qdrant = None
        self.collection = "bazi_knowledge"
        if settings.vector_backend == "qdrant" and vectors:
            self._initialize_qdrant(settings, documents, vectors)

    def _initialize_qdrant(
        self,
        settings: Settings,
        documents: list[RetrievalDocument],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client import QdrantClient, models

        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.qdrant = QdrantClient(path=str(settings.qdrant_path))
        dimension = len(vectors[0])
        if self.qdrant.collection_exists(self.collection):
            info = self.qdrant.get_collection(self.collection)
            configured = info.config.params.vectors.size
            if configured != dimension:
                self.qdrant.delete_collection(self.collection)
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
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

    def search(self, vector: list[float], allowed: set[str], limit: int) -> list[tuple[str, float]]:
        if self.qdrant is not None:
            response = self.qdrant.query_points(
                collection_name=self.collection,
                query=vector,
                limit=min(len(self.documents), max(limit * 4, limit)),
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


class LexicalReranker:
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
            scores.append(rank_prior * 0.9 + lexical * 0.1)
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers 未安装，请安装 requirements-local-ml.txt"
            ) from exc
        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[RetrievalDocument]) -> list[float]:
        pairs = [(query, f"{doc.title}\n{doc.text}") for doc in documents]
        return [float(value) for value in self.model.predict(pairs)]


class RetrievalService:
    def __init__(
        self,
        documents: list[RetrievalDocument],
        embedding_provider: EmbeddingProvider,
        bm25: BM25Index,
        vectors: VectorIndex,
        reranker: LexicalReranker | CrossEncoderReranker,
        settings: Settings,
    ) -> None:
        self.documents = {document.id: document for document in documents}
        self.embedding_provider = embedding_provider
        self.bm25 = bm25
        self.vectors = vectors
        self.reranker = reranker
        self.settings = settings

    @classmethod
    async def create(
        cls,
        settings: Settings,
        documents: list[RetrievalDocument],
        embedding_provider: EmbeddingProvider,
    ) -> RetrievalService:
        texts = [f"{doc.title}\n{doc.text}\n{' '.join(doc.concepts)}" for doc in documents]
        embeddings = await embedding_provider.embed(texts)
        reranker: LexicalReranker | CrossEncoderReranker
        if settings.reranker_provider == "cross_encoder":
            reranker = CrossEncoderReranker(settings.reranker_model)
        else:
            reranker = LexicalReranker()
        return cls(
            documents=documents,
            embedding_provider=embedding_provider,
            bm25=BM25Index(documents),
            vectors=VectorIndex(settings, documents, embeddings),
            reranker=reranker,
            settings=settings,
        )

    async def search(
        self,
        query: str,
        chart: ChartFacts,
        school: str,
        mode: str,
        limit: int = 6,
    ) -> list[RetrievalHit]:
        if mode == "lightrag":
            return await self._search_lightrag(query, limit)
        allowed = {
            document.id
            for document in self.documents.values()
            if self._eligible(document, chart, school)
        }
        query_vector = (await self.embedding_provider.embed([query]))[0]
        dense = self.vectors.search(query_vector, allowed, max(limit * 2, 10))
        if mode == "dense":
            return self._hits_from_single(dense, "dense", limit)
        sparse = self.bm25.search(query, allowed, max(limit * 2, 10))
        fused = self._rrf({"dense": dense, "bm25": sparse})
        hits = [
            RetrievalHit(
                document=self.documents[document_id],
                score=score,
                matched_by=sorted(components),
                component_scores=raw,
            )
            for document_id, score, components, raw in fused
        ]
        self._apply_concept_boost(query, hits)
        if mode == "hybrid_rerank" and hits:
            rerank_scores = self.reranker.score(query, [hit.document for hit in hits])
            for hit, rerank_score in zip(hits, rerank_scores, strict=True):
                hit.component_scores["reranker"] = rerank_score
                hit.score = hit.score * 0.35 + rerank_score * 0.65
                hit.matched_by.append("reranker")
            hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def _search_lightrag(self, query: str, limit: int) -> list[RetrievalHit]:
        if not self.settings.lightrag_base_url:
            raise RuntimeError("LightRAG 尚未配置，请设置 LIGHTRAG_BASE_URL")
        headers = {"Content-Type": "application/json"}
        if self.settings.lightrag_api_key:
            headers["X-API-Key"] = self.settings.lightrag_api_key
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.settings.lightrag_base_url.rstrip('/')}/query/data",
                headers=headers,
                json={
                    "query": query,
                    "mode": "mix",
                    "chunk_top_k": max(limit, 5),
                    "enable_rerank": True,
                },
            )
            response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("message", "LightRAG 查询失败"))
        data = payload.get("data", {})
        candidates: list[tuple[str, str, str]] = []
        for chunk in data.get("chunks", []):
            candidates.append(
                (
                    str(chunk.get("chunk_id", f"chunk-{len(candidates)}")),
                    str(chunk.get("file_path", "LightRAG chunk")),
                    str(chunk.get("content", "")),
                )
            )
        if not candidates:
            for entity in data.get("entities", []):
                candidates.append(
                    (
                        str(entity.get("entity_name", f"entity-{len(candidates)}")),
                        str(entity.get("file_path", "LightRAG entity")),
                        str(entity.get("description", "")),
                    )
                )
        hits: list[RetrievalHit] = []
        for rank, (item_id, source, text) in enumerate(candidates[:limit], start=1):
            document = RetrievalDocument(
                id=f"lightrag:{item_id}",
                kind="source_passage",
                title=source.rsplit("/", 1)[-1] or "LightRAG 证据",
                text=text,
                source=source,
                school="LightRAG",
                concepts=[],
            )
            hits.append(
                RetrievalHit(
                    document=document,
                    score=1 / rank,
                    matched_by=["lightrag-mix"],
                    component_scores={"lightrag": 1 / rank},
                )
            )
        return hits

    @staticmethod
    def _apply_concept_boost(query: str, hits: list[RetrievalHit]) -> None:
        for hit in hits:
            matched = [concept for concept in hit.document.concepts if concept in query]
            if not matched:
                continue
            bonus = min(0.02, 0.006 * len(matched))
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
        self, rankings: dict[str, list[tuple[str, float]]], k: int = 60
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
    def _eligible(document: RetrievalDocument, chart: ChartFacts, school: str) -> bool:
        if document.school not in {school, "基础共识"}:
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
