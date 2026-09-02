import asyncio
import secrets
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from bazi_api.core.config import Settings
from bazi_api.core.container import ApplicationContainer
from bazi_api.core.dependencies import get_container

from .schemas import FeedbackInput

router = APIRouter(tags=["feedback"])


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def create_feedback(
    payload: FeedbackInput,
    request: Request,
    container: ApplicationContainer = Depends(get_container),
) -> Response:
    client_key = resolve_feedback_client_key(request, container.settings)
    await asyncio.to_thread(
        container.feedback.add,
        str(payload.session_id),
        str(payload.message_id),
        payload.rating,
        payload.note,
        client_key,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def resolve_feedback_client_key(request: Request, settings: Settings) -> str:
    peer = request.client.host if request.client is not None else "unknown"
    secret = str(getattr(settings, "internal_proxy_secret", ""))
    networks = list(getattr(settings, "trusted_proxy_cidrs", []))
    if not secret or not networks:
        return peer

    try:
        peer_ip = ip_address(peer)
    except ValueError:
        return peer
    if not any(peer_ip in ip_network(value, strict=False) for value in networks):
        return peer

    supplied_secret = request.headers.get("X-Bazi-Proxy-Secret", "")
    forwarded = request.headers.get("X-Bazi-Client-IP", "").strip()
    try:
        forwarded_ip = ip_address(forwarded) if "," not in forwarded else None
    except ValueError:
        forwarded_ip = None
    if (
        forwarded_ip is None
        or not supplied_secret
        or not secrets.compare_digest(supplied_secret, secret)
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="代理身份无效")
    return forwarded_ip.compressed
