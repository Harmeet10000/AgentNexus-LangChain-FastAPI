from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from beanie import Document, Indexed, PydanticObjectId

# from bson import ObjectId
from pydantic import BaseModel, Field


class TransportMode(str, Enum):
    land = "land"
    sea = "sea"
    air = "air"


class Location(BaseModel):
    name: str
    coordinates: list[float]  # [longitude, latitude]


class RouteInfo(BaseModel):
    distance_km: float
    duration_hours: float
    co2_emissions_kg: float
    geometry: dict  # GeoJSON LineString


class Metadata(BaseModel):
    api_version: str
    calculation_method: str


class Search(Document):
    user_id: Annotated[PydanticObjectId, Indexed()]
    origin: Location
    destination: Location
    cargo_weight_kg: float
    transport_mode: TransportMode
    shortest_route: RouteInfo
    efficient_route: RouteInfo
    metadata: Metadata
    created_at: Annotated[datetime, Indexed()] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Settings:
        name = "searches"
