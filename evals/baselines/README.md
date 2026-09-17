# Retrieval Baselines

`questions.bm25.json` pins the dataset, retrieval index, model, settings, and
measured scores for the CI quality gate. An identity mismatch must fail;
changing the index version alone does not validate a replacement baseline.

## Baseline identity

| Field | Value |
| --- | --- |
| `mode` | `bm25` |
| `model_version` | `bm25-lexicon-v1` |
| `index_version` | `35c40017a256c023` |
| `dataset` | `questions.json` (61 cases) |
| `dataset_sha256` | `bd97eb14beb5fcd9e8e5c34bf963d77c8d9a50fd0bd094c58f066c5cac90e051` |
| `limit` | `10` |
| `evidence_scope` | `reviewed_only` |
| `answer_mode` | `off` |
| `max_regression` | `0.02` |

## Measured scores (2026-09-17)

| Evaluation | Recall@5 | MRR@10 |
| --- | --- | --- |
| `questions.json` (61 cases, this baseline) | 0.7869 | 0.5998 |
| `multi-turn.json` (10 cases, no baseline file) | 0.9000 | 0.6950 |

`evidence_text_consistency`, `evidence_id_resolution`, and `citation_chain_rate`
remain `1.0` on the main question set. Thirteen cases miss: `q004`, `q006`,
`q007`, `q014`, `q018`, `q028`, `q029`, `q032`, `q035`, `q036`, `q046`,
`q056`, `q058`. They are kept in the baseline report; expected evidence IDs and
scoring rules were not adjusted to pass them.

## Why the acceptance thresholds were lowered

The previous `hybrid` baseline recorded `Recall@5 = 0.9344` / `MRR@10 = 0.8864`
against the local vector stack (hash embeddings plus a lexical reranker) and
enforced absolute gates of `Recall@5 >= 0.90` and `MRR@10 >= 0.80`.

That stack was deleted by design. The main answer path no longer retrieves at
all: the model answers directly from the computed chart and topic facts. The
knowledge base is now an **on-demand verification channel** — a BM25-only
lookup used to check a specific term or passage when the user asks for one —
so retrieval quality is measured against a deliberately smaller, non-vector
index and the old 0.90/0.80 gates are unreachable rather than regressed.

The gates are re-based on the measured BM25 numbers with margin instead of
being dropped:

| Gate | Old | New | Measured |
| --- | --- | --- | --- |
| `recall_at_5` | 0.90 | 0.75 | 0.7869 |
| `mrr_at_10` | 0.80 | 0.55 | 0.5998 |
| `multi_turn_recall_at_5` | 0.90 | 0.85 | 0.9000 |

`evidence_text_consistency == 1.0` and `evidence_id_resolution == 1.0` are
unchanged: citation identifiers must still resolve and quoted passages must
still match the source text.

## Reproduction

Run from the backend directory:

```bash
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode bm25 --limit 10 \
  --output evals/baselines/questions.bm25.json
```

CI compares against the baseline and enforces acceptance:

```bash
PYTHONPATH=src uv run python -m bazi_api.cli.evaluate --mode bm25 --limit 10 \
  --require-acceptance --baseline evals/baselines/questions.bm25.json \
  --output evaluation-report.json
```

For a deliberate corpus or configuration migration, first generate a candidate
report without `--baseline`, inspect every changed score and miss, and record
the reason for the new reference. Do not raise the regression tolerance, remove
identity checks, or discard failing cases to make CI green. GitHub Actions
uploads the generated report as `offline-retrieval-report` for inspection.
