import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bazi_api.core.config import Settings

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "prewarm_retrieval.py"
SPEC = importlib.util.spec_from_file_location("prewarm_retrieval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prewarm_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prewarm_module)


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_prewarm_does_not_load_reranker_and_always_closes(
    monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    retrieval = SimpleNamespace(
        documents=[object()],
        vectors=SimpleNamespace(documents=[object()]),
        model_version="hash-fallback" if fallback else "expected-model",
        index_version="index",
        cache_stats={"hits": 0, "misses": 1},
        search=AsyncMock(),
        close=Mock(),
    )
    monkeypatch.setattr(
        prewarm_module.RetrievalService, "create", AsyncMock(return_value=retrieval)
    )
    monkeypatch.setattr(
        prewarm_module,
        "create_embedding_provider",
        lambda *args: SimpleNamespace(model_version="expected-model"),
    )
    repository = SimpleNamespace(documents=lambda scope: [], graph_nodes=[], graph_edges=[])
    if fallback:
        with pytest.raises(RuntimeError, match="cache is not ready"):
            await prewarm_module.prewarm(Settings(_env_file=None), repository, Mock())
    else:
        await prewarm_module.prewarm(Settings(_env_file=None), repository, Mock())
    retrieval.search.assert_not_called()
    retrieval.close.assert_called_once_with()
