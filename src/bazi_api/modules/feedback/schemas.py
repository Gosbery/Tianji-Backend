from typing import Literal

from pydantic import UUID4, BaseModel, Field


class FeedbackInput(BaseModel):
    session_id: UUID4
    message_id: UUID4
    rating: Literal[-1, 1]
    note: str = Field(default="", max_length=1000)
