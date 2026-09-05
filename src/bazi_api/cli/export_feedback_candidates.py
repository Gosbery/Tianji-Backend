from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def export_candidates(database_path: Path) -> list[dict[str, Any]]:
    if not database_path.exists():
        raise ValueError(f"数据库不存在: {database_path}")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT f.id AS feedback_id, f.note, f.created_at,
                   a.id AS message_id, a.content AS answer, a.payload_json,
                   u.content AS question, s.id AS session_id,
                   s.chart_fingerprint, s.school, s.evidence_scope,
                   t.mode, t.index_version, t.generation_model,
                   t.prompt_version, t.question_policy, t.policy_decision,
                   t.citations_validated, t.degradation_reason, t.hits_json
            FROM feedback AS f
            JOIN messages AS a
              ON a.id = f.message_id
             AND a.session_id = f.session_id
             AND a.role = 'assistant'
            JOIN sessions AS s ON s.id = f.session_id
            LEFT JOIN messages AS u
              ON u.session_id = a.session_id
             AND u.turn_index = a.turn_index
             AND u.role = 'user'
            LEFT JOIN retrieval_traces AS t ON t.message_id = a.id
            WHERE f.rating = -1
            ORDER BY f.created_at, f.id
            """
        ).fetchall()

    candidates = []
    for row in rows:
        payload = _json_object(row["payload_json"])
        hits = _json_list(row["hits_json"])
        candidates.append(
            {
                "candidate_id": f"feedback:{row['feedback_id']}",
                "review_status": "pending_human_review",
                "question": row["question"] or "",
                "answer": row["answer"],
                "feedback_note": row["note"],
                "session_context": {
                    "chart_fingerprint": row["chart_fingerprint"],
                    "school": row["school"],
                    "evidence_scope": row["evidence_scope"],
                },
                "generation": {
                    "mode": row["mode"] or payload.get("mode", ""),
                    "index_version": row["index_version"] or "",
                    "model": row["generation_model"] or "",
                    "prompt_version": row["prompt_version"] or "",
                    "question_policy": row["question_policy"] or "",
                    "policy_decision": row["policy_decision"]
                    or payload.get("policy_decision", ""),
                    "citations_validated": bool(row["citations_validated"]),
                    "degradation_reason": row["degradation_reason"] or "",
                },
                "evidence_ids": [
                    str(item["document"]["id"])
                    for item in hits
                    if isinstance(item, dict)
                    and isinstance(item.get("document"), dict)
                    and item["document"].get("id")
                ],
                "feedback_created_at": row["created_at"],
                "source": {
                    "session_id": row["session_id"],
                    "message_id": row["message_id"],
                },
            }
        )
    return candidates


def _json_object(value: object) -> dict[str, Any]:
    parsed = _json_value(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: object) -> list[Any]:
    parsed = _json_value(value, [])
    return parsed if isinstance(parsed, list) else []


def _json_value(value: object, fallback: object) -> object:
    if not isinstance(value, str):
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出负反馈为待人工确认的候选评测案例"
    )
    parser.add_argument("--database", type=Path, default=Path("data/app.db"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        candidates = export_candidates(args.database)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(2, f"export-feedback-candidates: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已导出 {len(candidates)} 条待人工确认的候选案例到 {args.output}")


if __name__ == "__main__":
    main()
