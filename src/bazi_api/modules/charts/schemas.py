from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BirthInput(BaseModel):
    date: date
    time: time
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


class ChartFacts(BaseModel):
    calculated_at: datetime
    birth: BirthInput
    pillars: list[PillarFacts]
    day_master: str
    day_master_element: str
    day_master_yin_yang: str
    calculation_basis: str
    uncertainties: list[str] = Field(default_factory=list)
