from datetime import datetime

from pydantic import BaseModel


class Pagination(BaseModel):
    page: int
    limit: int
    total: int
    total_pages: int
    has_next: bool


class SearchOut(BaseModel):
    id: str
    origin: dict
    destination: dict
    cargo_weight_kg: float
    transport_mode: str
    shortest_route: dict
    efficient_route: dict
    created_at: datetime


class SearchListResponse(BaseModel):
    data: list[SearchOut]
    pagination: Pagination


class SearchStatsResponse(BaseModel):
    total_searches: int
    total_co2_saved: float
    avg_cargo_weight: float
