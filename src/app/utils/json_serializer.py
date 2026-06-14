"""Reusable JSON serialization via Pydantic TypeAdapter.

Replaces orjson. Uses module-level TypeAdapter singletons to avoid
recreating adapters on every call.
"""

from typing import Any

from pydantic import TypeAdapter

_adapter_any: TypeAdapter[Any] = TypeAdapter(Any)
_adapter_dict: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_adapter_float_list: TypeAdapter[list[float]] = TypeAdapter(list[float])


def to_json_bytes(data: object) -> bytes:
    return _adapter_any.dump_json(data)


def to_json_str(data: object) -> str:
    return _adapter_any.dump_json(data).decode("utf-8")


def to_sorted_key_bytes(data: dict[str, object]) -> bytes:
    return _adapter_dict.dump_json(dict(sorted(data.items())))


def to_float_list_bytes(data: list[float]) -> bytes:
    return _adapter_float_list.dump_json(data)


def to_float_list_str(data: list[float]) -> str:
    return _adapter_float_list.dump_json(data).decode("utf-8")


def from_json(data: str | bytes) -> object:
    return _adapter_any.validate_json(data)


def from_json_float_list(data: str | bytes) -> list[float]:
    return _adapter_float_list.validate_json(data)
