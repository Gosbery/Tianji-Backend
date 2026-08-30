from fastapi import Request

from .container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container
