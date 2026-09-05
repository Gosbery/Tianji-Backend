from pathlib import Path

from bazi_api.cli.export_feedback_candidates import export_candidates
from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.feedback.repository import FeedbackRepository
from bazi_api.modules.observability.repository import TraceRepository


def test_export_only_negative_feedback_as_pending_candidates(tmp_path: Path) -> None:
    database_path = tmp_path / "app.db"
    database = SQLiteDatabase(database_path)
    conversations = ConversationRepository(database)
    try:
        session_id, is_new = conversations.resolve_session(
            chart_fingerprint="chart-a",
            school="基础共识",
            evidence_scope="reviewed_only",
        )
        with database.transaction() as connection:
            _, message_id = conversations.save_exchange(
                connection,
                session_id=session_id,
                is_new_session=is_new,
                question="如何理解日主？",
                answer="回答 [1]",
                chart_fingerprint="chart-a",
                school="基础共识",
                evidence_scope="reviewed_only",
                assistant_payload={"mode": "hybrid"},
            )
            TraceRepository(database).add(
                session_id=session_id,
                question="如何理解日主？",
                mode="hybrid",
                evidence_scope="reviewed_only",
                model_version="hash-test",
                index_version="index-test",
                hits=[{"document": {"id": "concept-day-master"}}],
                latency_ms=10,
                token_usage=12,
                message_id=message_id,
                generation_model="model-test",
                prompt_version="prompt-test",
                question_policy="evidence_answer",
                policy_decision="allow",
                citations_validated=True,
                degradation_reason="",
                connection=connection,
            )
        FeedbackRepository(database).add(session_id, message_id, -1, "引用不够清楚")

        candidates = export_candidates(database_path)

        assert len(candidates) == 1
        assert candidates[0]["review_status"] == "pending_human_review"
        assert candidates[0]["question"] == "如何理解日主？"
        assert candidates[0]["evidence_ids"] == ["concept-day-master"]
        assert candidates[0]["generation"]["prompt_version"] == "prompt-test"
    finally:
        database.close()
