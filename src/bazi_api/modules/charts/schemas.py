from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BirthInput(BaseModel):
    date: date
    time: time
    gender: Literal["male", "female"] = "male"
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=64)
    name: str = Field(default="访客", max_length=100)

    @field_validator("date")
    @classmethod
    def supported_date(cls, value: date) -> date:
        if not 1900 <= value.year <= 2100:
            raise ValueError("首版支持 1900 至 2100 年的出生日期")
        return value

    @field_validator("timezone")
    @classmethod
    def supported_timezone(cls, value: str) -> str:
        if value != "Asia/Shanghai":
            raise ValueError("首版排盘仅支持 Asia/Shanghai 时区")
        return value


class PillarFacts(BaseModel):
    key: Literal["year", "month", "day", "time"]
    label: str
    stem: str
    branch: str
    stem_element: str
    stem_yin_yang: str
    branch_element: str
    hidden_stems: list[str]
    ten_god_stem: str
    ten_god_branches: list[str]
    nayin: str


class MonthCommandFacts(BaseModel):
    branch: str
    main_hidden_stem: str
    ten_god: str


class ExposedStemFact(BaseModel):
    hidden_stem: str
    hidden_positions: list[str]
    visible_positions: list[str]
    outside_day_master: bool


class RootFact(BaseModel):
    stem: str
    stem_position: str
    branch: str
    branch_position: str
    hidden_stem: str
    relation: Literal["same_stem", "same_element"]


class StructuralRelationFact(BaseModel):
    kind: Literal[
        "stem_combination",
        "branch_combination",
        "branch_clash",
        "branch_harm",
        "branch_punishment",
        "three_harmony",
        "seasonal_meeting",
    ]
    label: str
    members: list[str]
    positions: list[str]
    completeness: Literal["pair", "partial", "full"] = "pair"
    note: str = ""


class PatternCandidateFact(BaseModel):
    name: str
    ten_god: str
    source_stem: str
    basis: Literal["month_main_qi", "month_hidden_stem_exposed"]
    exposed_positions: list[str] = Field(default_factory=list)
    note: str


class LuckCycleFact(BaseModel):
    index: int = Field(ge=1)
    ganzhi: str
    stem: str
    branch: str
    start_year: int
    end_year: int
    start_age: int
    end_age: int
    status: Literal["past", "current", "future"]


class LuckFacts(BaseModel):
    direction: Literal["forward", "reverse"]
    direction_label: Literal["顺排", "逆排"]
    start_at: datetime
    start_offset_years: int = Field(ge=0)
    start_offset_months: int = Field(ge=0, le=11)
    start_offset_days: int = Field(ge=0)
    start_offset_hours: int = Field(ge=0)
    method: str
    cycles: list[LuckCycleFact]
    current_cycle: LuckCycleFact | None = None
    next_cycle: LuckCycleFact | None = None


class ChartFacts(BaseModel):
    calculated_at: datetime
    birth: BirthInput
    pillars: list[PillarFacts]
    day_master: str
    day_master_element: str
    day_master_yin_yang: str
    month_command: MonthCommandFacts
    exposed_stems: list[ExposedStemFact] = Field(default_factory=list)
    roots: list[RootFact] = Field(default_factory=list)
    structural_relations: list[StructuralRelationFact] = Field(default_factory=list)
    pattern_candidates: list[PatternCandidateFact] = Field(default_factory=list)
    luck: LuckFacts
    calculation_basis: str
    uncertainties: list[str] = Field(default_factory=list)
