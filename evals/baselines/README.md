# Offline Retrieval Baselines

`questions.hash.hybrid.json` pins the dataset, retrieval index, model, settings,
and measured scores for the CI quality gate. An identity mismatch must fail;
changing the index version alone does not validate a replacement baseline.

## 2026-09-09 Migration

The previous baseline used index `76a1a330ce14d5e7` and 627 indexed documents
(625 cache hits plus 2 misses). The expanded corpus contains 3,439 documents.
The fixed question set remains 61 cases with digest
`4abc4c1ca341313ebdc869c856e9eab126e295b79553f71d6b0efdcef2dc7ce7`.

| Evaluation | Index | Recall@5 | MRR@10 |
| --- | --- | --- | --- |
| Historical smaller corpus | `76a1a330ce14d5e7` | 1.0000 | 0.9508 |
| Parent retrieval implementation (`38a903d`), current corpus | `7229c7a012eb8446` | 0.9344 | 0.8864 |
| Current implementation (`31382b2`), current corpus | `7bc23bc53eeff731` | 0.9344 | 0.8864 |

The comparison of parent and current retrieval implementations held the current
data, query preparation, settings, hash embeddings, offline answer generation,
and evaluation code constant. The current implementation was also evaluated
again before refreshing this baseline. It introduces no additional misses in
this comparison. Both the parent and current CI runs failed against the obsolete
index identity: GitHub Actions runs `33937493549` and `34330599084`.

The historical score loss is real and is not a claim of improved quality. This
baseline migration records the expanded corpus's measured state. Cases `q004`,
`q049`, `q055`, and `q060` remain misses and are retained in the baseline report.
They concern single-element inference limits, school differences, and day-change
rules. The historical report remains available in Git history before this update.
Neither expected evidence IDs nor scoring rules were changed to pass these cases.

The maximum regression tolerance stays at 0.02. CI also explicitly enforces the
existing absolute acceptance checks, including Recall@5 >= 0.90 and MRR@10 >= 0.80.
Citation checks remain enabled. These measure offline evidence retrieval and a
source-directory answer, not the factual accuracy of live LLM answers.

## Reproduction

Run from the backend directory with the development and retrieval dependencies:

```bash
EMBEDDING_PROVIDER=hash RERANKER_PROVIDER=lexical VECTOR_BACKEND=memory PYTHONPATH=src \
  uv run python -m bazi_api.cli.evaluate --mode hybrid --limit 10 \
  --answer-mode offline --require-acceptance \
  --baseline evals/baselines/questions.hash.hybrid.json \
  --output /tmp/bazi-offline-evaluation.json
```

For a deliberate corpus or retrieval configuration migration, first generate a
candidate report without `--baseline`, inspect every changed score and miss, and
record the reason for the new reference. Do not raise the regression tolerance,
remove identity checks, or discard failing cases to make CI green. GitHub Actions
uploads the generated report as `offline-retrieval-report` for inspection.
