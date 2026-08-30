from pathlib import Path

from bazi_api.modules.knowledge.repository import KnowledgeRepository

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def test_seed_knowledge_counts() -> None:
    repository = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    repository.load()

    assert len(repository.cards) == 50
    assert all(card.status == "reviewed" for card in repository.cards)
    assert len(repository.passages) >= 10
    assert len(repository.documents()) > len(repository.cards)
