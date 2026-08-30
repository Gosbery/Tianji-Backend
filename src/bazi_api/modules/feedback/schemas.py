from typing import Literal

from pydantic import BaseModel, Field


class FeedbackInput(BaseModel):
    session_id: str
    message_id: str | None = None
    rating: Literal[-1, 1]
    note: str = Field(default="", max_length=1000)
