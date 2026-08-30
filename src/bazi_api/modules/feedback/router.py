from fastapi import APIRouter, Depends, Response, status

from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import FeedbackInput

router = APIRouter(tags=["feedback"])


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def create_feedback(
    payload: FeedbackInput,
    container: ApplicationContainer = Depends(get_container),
) -> Response:
    container.feedback.add(
        payload.session_id,
        payload.message_id,
        payload.rating,
        payload.note,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
