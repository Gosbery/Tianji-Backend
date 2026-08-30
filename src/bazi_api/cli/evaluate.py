from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict

from bazi_api.core.config import get_settings
from bazi_api.integrations.embeddings import create_embedding_provider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.retrieval.service import RetrievalService


async def run(mode: str, limit: int) -> dict[str, object]:
    settings = get_settings()
    settings.ensure_directories()
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    provider = create_embedding_provider(settings)
    retrieval = await RetrievalService.create(settings, repository.documents(), provider)
    chart = ChartCalculator().calculate(
        BirthInput.model_validate({"date": "1990-01-01", "time": "12:00:00", "name": "评测样例"})
    )
    cases = json.loads((settings.evals_path / "questions.json").read_text(encoding="utf-8"))
    hits = 0
    per_category: defaultdict[str, list[int]] = defaultdict(list)
    misses = []
    for case in cases:
        results = await retrieval.search(case["question"], chart, "基础共识", mode, limit)
        retrieved = {result.document.id for result in results}
        success = bool(retrieved & set(case["expected_ids"]))
        hits += int(success)
        per_category[case["category"]].append(int(success))
        if not success:
            misses.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected": case["expected_ids"],
                    "retrieved": [result.document.id for result in results],
                }
            )
    return {
        "mode": mode,
        "cases": len(cases),
        "recall_at_k": round(hits / len(cases), 4),
        "by_category": {
            category: round(sum(values) / len(values), 4)
            for category, values in per_category.items()
        },
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="hybrid", choices=["dense", "hybrid", "hybrid_rerank", "lightrag"]
    )
    parser.add_argument("--limit", default=5, type=int)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.mode, args.limit)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
