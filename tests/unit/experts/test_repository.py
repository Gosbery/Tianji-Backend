from pathlib import Path

import pytest

from bazi_api.core.config import BACKEND_ROOT
from bazi_api.modules.experts.repository import ExpertRepository
from bazi_api.modules.knowledge.repository import KnowledgeRepository


def test_expert_catalog_rejects_unknown_method_card(tmp_path: Path) -> None:
    knowledge = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    knowledge.load()
    root = tmp_path / "experts"
    root.mkdir()
    (root / "invalid.yml").write_text(
        """
schema_version: 1
experts:
  - id: invalid
    display_name: 无效方法
    version: '1'
    review_status: machine_verified
    allowed_schools: [基础共识]
    method_cards: [{card_id: missing-card, purpose: test}]
    methodology: [test]
    boundaries: [test]
    is_default: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown method card: missing-card"):
        ExpertRepository(root, knowledge).load()


def test_machine_verified_experts_are_preview_only() -> None:
    knowledge = KnowledgeRepository(BACKEND_ROOT / "knowledge")
    knowledge.load()
    experts = ExpertRepository(BACKEND_ROOT / "knowledge" / "experts", knowledge)
    experts.load()

    assert [item.id for item in experts.list()] == ["comprehensive"]
    assert {item.id for item in experts.list("personal_preview")} == {
        "comprehensive",
        "liang-xiangrun",
        "duan-jianye",
    }
    assert experts.default_expert_id == "liang-xiangrun"
    prompt = experts.prompt_context(experts.get("liang-xiangrun"))
    assert "不模仿人物口吻" in prompt
    assert "自称" in prompt
