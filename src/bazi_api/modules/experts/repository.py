from __future__ import annotations

from pathlib import Path

import yaml

from bazi_api.modules.knowledge.repository import KnowledgeRepository

from .schemas import ExpertCatalog, ExpertProfile


class ExpertRepository:
    def __init__(self, root: Path, knowledge: KnowledgeRepository) -> None:
        self.root = root
        self.knowledge = knowledge
        self._experts: dict[str, ExpertProfile] = {}

    def load(self) -> None:
        profiles: list[ExpertProfile] = []
        for path in sorted(self.root.glob("*.y*ml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            catalog = ExpertCatalog.model_validate(payload)
            profiles.extend(catalog.experts)
        if not profiles:
            raise RuntimeError(f"No expert profiles found under {self.root}")
        ids = [profile.id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate expert ids")
        defaults = [profile.id for profile in profiles if profile.is_default]
        if len(defaults) != 1:
            raise ValueError("Expert catalog requires exactly one default expert")
        known_cards = {card.id: card for card in self.knowledge.cards}
        for profile in profiles:
            for reference in profile.method_cards:
                card = known_cards.get(reference.card_id)
                if card is None:
                    raise ValueError(
                        f"Expert {profile.id} references unknown method card: {reference.card_id}"
                    )
                if card.card_type != "method":
                    raise ValueError(
                        f"Expert {profile.id} references non-method card: {reference.card_id}"
                    )
                if profile.review_status == "reviewed" and card.status != "reviewed":
                    raise ValueError(
                        f"Reviewed expert {profile.id} cannot reference {card.status} card: "
                        f"{reference.card_id}"
                    )
        self._experts = {profile.id: profile for profile in profiles}

    @property
    def default_expert_id(self) -> str:
        return next(profile.id for profile in self._experts.values() if profile.is_default)

    def get(self, expert_id: str) -> ExpertProfile:
        try:
            return self._experts[expert_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"Unknown expert: {expert_id}") from exc

    def list(self, scope: str = "reviewed_only") -> list[ExpertProfile]:
        allowed = {"reviewed"}
        if scope == "personal_preview":
            allowed.add("machine_verified")
        return [
            profile.model_copy(deep=True)
            for profile in self._experts.values()
            if profile.review_status in allowed
        ]

    def prompt_context(self, profile: ExpertProfile) -> str:
        cards = {card.id: card for card in self.knowledge.cards}
        steps = "\n".join(f"- {item}" for item in profile.methodology)
        boundaries = "\n".join(f"- {item}" for item in profile.boundaries)
        card_rules = "\n".join(
            f"- {reference.purpose}：{cards[reference.card_id].content}"
            for reference in profile.method_cards
        )
        return (
            f"当前采用“{profile.display_name}”（配置版本 {profile.version}）的方法框架。\n"
            "只采用公开材料中可审核的方法，不模仿人物口吻，不自称或暗示自己是该人物。\n"
            f"分析顺序：\n{steps}\n方法卡：\n{card_rules or '- 无'}\n适用边界：\n{boundaries}"
        )
