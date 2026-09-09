import base64
import binascii
import hashlib
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import Settings

ACCESS_KEY_HEADER = "x-bazi-access-key"
ACCESS_USERNAME = "bazi"


def application_access_error(request: Request, settings: Settings) -> JSONResponse | None:
    expected = settings.app_access_key
    if not expected:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "请先配置应用访问口令 APP_ACCESS_KEY",
                "code": "access_unconfigured",
            },
            headers={"Cache-Control": "no-store"},
        )

    supplied = _access_key(request)
    if supplied is not None and secrets.compare_digest(
        hashlib.sha256(supplied.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    ):
        if _cross_site_write(request, settings):
            return JSONResponse(
                status_code=403,
                content={"detail": "不允许跨站修改数据", "code": "cross_site_request"},
                headers={"Cache-Control": "no-store"},
            )
        return None
    return JSONResponse(
        status_code=401,
        content={"detail": "需要应用访问凭证", "code": "authentication_required"},
        headers={
            "WWW-Authenticate": 'Basic realm="BaZi", charset="UTF-8"',
            "Cache-Control": "no-store",
        },
    )


def _access_key(request: Request) -> str | None:
    keys = request.headers.getlist(ACCESS_KEY_HEADER)
    if keys:
        return keys[0] if len(keys) == 1 else None
    authorizations = request.headers.getlist("authorization")
    if len(authorizations) != 1:
        return None
    scheme, _, value = authorizations[0].partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    username, separator, password = decoded.partition(":")
    if not separator or username != ACCESS_USERNAME:
        return None
    return password


def _cross_site_write(request: Request, settings: Settings) -> bool:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return False
    if request.headers.get("sec-fetch-site") == "cross-site":
        return True
    origin = request.headers.get("origin")
    if origin is None:
        return False
    own_origin = f"{request.url.scheme}://{request.url.netloc}"
    return origin != own_origin and origin not in settings.cors_origins
