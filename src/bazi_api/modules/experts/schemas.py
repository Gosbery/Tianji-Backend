from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from bazi_api.modules.knowledge.schemas import ReviewStatus


class ExpertMethodCard(BaseModel):
    card_id: str
    purpose: str


class ExpertProfile(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    display_name: str
    version: str
    review_status: ReviewStatus
    allowed_schools: list[str] = Field(min_length=1)
    preferred_schools: list[str] = Field(default_factory=list)
    method_cards: list[ExpertMethodCard] = Field(default_factory=list)
    boundaries: list[str] = Field(min_length=1)
    methodology: list[str] = Field(min_length=1)
    attribution: str = ""
    is_default: bool = False

    @model_validator(mode="after")
    def validate_school_preferences(self) -> ExpertProfile:
        unknown = set(self.preferred_schools).difference(self.allowed_schools)
        if unknown:
            raise ValueError(f"preferred_schools not allowed: {', '.join(sorted(unknown))}")
        return self


class ExpertCatalog(BaseModel):
    schema_version: Literal[1] = 1
    experts: list[ExpertProfile]


class ExpertList(BaseModel):
    experts: list[ExpertProfile]
    default_expert_id: str
