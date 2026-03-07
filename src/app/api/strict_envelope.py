"""Global API response envelope enforcement for routers."""

from __future__ import annotations

import os
from typing import get_origin

from fastapi import APIRouter

from app.shared.response_type import APIResponse
from app.utils import logger


class StrictEnvelopeAPIRouter(APIRouter):
    """APIRouter that validates `response_model` uses `APIResponse[T]`."""

    strict_enforce: bool = os.getenv("STRICT_ENVELOPE_ENFORCE", "false").lower() == "true"

    def add_api_route(self, path: str, endpoint, **kwargs) -> None:
        response_model = kwargs.get("response_model")

        if self._is_envelope_violation(response_model=response_model):
            endpoint_name = getattr(endpoint, "__name__", "unknown")
            message = (
                f"Route '{path}' ({endpoint_name}) must declare "
                "response_model=APIResponse[T]"
            )
            if self.strict_enforce:
                raise ValueError(message)
            # logger.warning(message)

        super().add_api_route(path, endpoint, **kwargs)

    @staticmethod
    def _is_envelope_violation(response_model) -> bool:
        if response_model is None:
            return True
        return get_origin(response_model) is not APIResponse
