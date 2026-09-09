import asyncio
import math
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)


@pytest.mark.asyncio
async def test_sentence_transformer_encoding_runs_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SentenceTransformerEmbeddingProvider("test-model")
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_encode(_: list[str]) -> list[list[float]]:
        worker_threads.append(threading.get_ident())
        return [[1.0, 0.0]]

    monkeypatch.setattr(provider, "_encode", fake_encode)

    assert await provider.embed(["文本"]) == [[1.0, 0.0]]
    assert worker_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_hash_embedding_batch_runs_off_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HashEmbeddingProvider()
    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_vector(_: str) -> list[float]:
        worker_threads.append(threading.get_ident())
        return [1.0]

    monkeypatch.setattr(provider, "_vector", fake_vector)

    assert await provider.embed(["文本"]) == [[1.0]]
    assert worker_threads[0] != event_loop_thread


class FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def to(self, _: str) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[list[float]]:
        return self.values


class FakeTokenizer:
    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

    def __call__(self, text: str, **options: object) -> dict[str, object]:
        assert options["truncation"] is True
        assert options["return_overflowing_tokens"] is True
        assert options["padding"] is False
        tokens = [int(value) for value in text.split()]
        budget = int(options["max_length"]) - 2
        step = budget - int(options["stride"])
        windows = []
        for start in range(0, max(1, len(tokens)), step):
            end = min(start + budget, len(tokens))
            windows.append([-101, *tokens[start:end], -102])
            if end == len(tokens):
                break
        return {
            "input_ids": windows,
            "attention_mask": [[1] * len(window) for window in windows],
            "overflow_to_sample_mapping": [0] * len(windows),
        }

    def pad(self, features: list[dict[str, object]], **_: object) -> dict[str, FakeTensor]:
        return {"input_ids": FakeTensor([item["input_ids"] for item in features])}  # type: ignore[list-item]


class FakeSentenceTransformer:
    def __init__(self, name: str, **options: object) -> None:
        self.name = name
        self.options = options
        self.tokenizer = FakeTokenizer()
        self.batches: list[list[list[float]]] = []
        self.eval_called = False

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, features: dict[str, FakeTensor]) -> dict[str, FakeTensor]:
        batch = features["input_ids"].values
        self.batches.append(batch)
        content = [[value for value in row if value >= 0] for row in batch]
        return {
            "sentence_embedding": FakeTensor(
                [[sum(row), row[-1] if row else 1.0] for row in content]
            )
        }


@pytest.fixture
def fake_model_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))


@pytest.mark.asyncio
async def test_local_embedding_bounds_batches_and_covers_the_entire_document(
    fake_model_runtime: None,
) -> None:
    provider = SentenceTransformerEmbeddingProvider(
        "test-model", local_files_only=True, batch_size=2, max_seq_length=6, window_overlap=1
    )

    vectors = await provider.embed(["1 2 3 4 5 6 7 8 9 10"])

    model = provider.model
    assert model.options == {"device": "cpu", "local_files_only": True}
    assert model.eval_called
    assert model.max_seq_length == 6
    assert [len(batch) for batch in model.batches] == [2, 1]
    windows = [row[1:-1] for batch in model.batches for row in batch]
    assert windows == [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]
    assert all(len(row) <= 6 for batch in model.batches for row in batch)
    weighted = [0.0, 0.0]
    for window, weight in zip(windows, [4, 3, 3], strict=True):
        vector = [sum(window), window[-1]]
        norm = math.hypot(*vector)
        weighted = [total + value / norm * weight for total, value in zip(weighted, vector)]
    norm = math.hypot(*weighted)
    assert vectors[0] == pytest.approx([value / norm for value in weighted])
    prefix_norm = math.hypot(10, 4)
    assert vectors[0] != pytest.approx([10 / prefix_norm, 4 / prefix_norm])


@pytest.mark.asyncio
async def test_embedding_preserves_document_order_and_handles_empty_text(
    fake_model_runtime: None,
) -> None:
    provider = SentenceTransformerEmbeddingProvider(
        "test-model", batch_size=2, max_seq_length=6, window_overlap=1
    )
    assert await provider.embed([]) == []
    assert provider.model is None

    vectors = await provider.embed(["1 2", "", "3 4"])

    assert len(vectors) == 3
    assert [len(batch) for batch in provider.model.batches] == [2, 1]
    assert vectors[0] == pytest.approx([3 / math.sqrt(13), 2 / math.sqrt(13)])
    assert vectors[1] == [0.0, 1.0]
    assert vectors[2] == pytest.approx([7 / math.sqrt(65), 4 / math.sqrt(65)])


def test_embedding_cache_version_tracks_window_semantics_but_not_batch_size() -> None:
    default = SentenceTransformerEmbeddingProvider("test-model")
    smaller_batch = SentenceTransformerEmbeddingProvider("test-model", batch_size=1)
    longer = SentenceTransformerEmbeddingProvider("test-model", max_seq_length=1024)
    different_overlap = SentenceTransformerEmbeddingProvider("test-model", window_overlap=32)
    assert default.model_version == smaller_batch.model_version
    assert len({default.model_version, longer.model_version, different_overlap.model_version}) == 3
    assert default.model_version != "sentence-transformers:test-model"


def test_embedding_factory_applies_local_resource_limits() -> None:
    settings = Settings(
        _env_file=None,
        local_embedding_device="cpu",
        local_embedding_batch_size=1,
        local_embedding_max_seq_length=256,
        local_embedding_window_overlap=32,
    )
    provider = create_embedding_provider(settings)
    assert provider.device == "cpu"
    assert provider.batch_size == 1
    assert provider.max_seq_length == 256
    assert provider.window_overlap == 32
    assert provider.model is None
    with pytest.raises(ValueError, match="overlap"):
        Settings(
            _env_file=None, local_embedding_max_seq_length=64, local_embedding_window_overlap=63
        )


@pytest.mark.asyncio
async def test_concurrent_queries_do_not_reload_a_failed_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = []

    def failed_model(*_: object, **__: object) -> None:
        loads.append(1)
        raise RuntimeError("model allocation failed")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=failed_model)
    )
    provider = SentenceTransformerEmbeddingProvider("test-model")

    results = await asyncio.gather(
        *(provider.embed(["test"]) for _ in range(3)), return_exceptions=True
    )

    assert len(loads) == 1
    assert all(isinstance(result, RuntimeError) for result in results)
    assert sum("disabled" in str(result) for result in results) == 2
    assert provider.model is None


@pytest.mark.asyncio
async def test_inference_failure_releases_model_and_disables_retries(
    fake_model_runtime: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = []

    class FailingModel(FakeSentenceTransformer):
        def __init__(self, name: str, **options: object) -> None:
            super().__init__(name, **options)
            loads.append(1)

        def __call__(self, _: object) -> object:
            raise MemoryError("inference allocation failed")

    monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", FailingModel)
    provider = SentenceTransformerEmbeddingProvider("test-model")
    with pytest.raises(MemoryError, match="inference allocation failed"):
        await provider.embed(["1 2"])
    assert provider.model is None
    with pytest.raises(RuntimeError, match="disabled"):
        await provider.embed(["1 2"])
    assert len(loads) == 1
