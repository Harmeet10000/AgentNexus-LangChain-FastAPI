import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection


class APIFeatures:
    """MongoDB query builder with filtering, sorting, and pagination."""

    def __init__(
        self, collection: AsyncIOMotorCollection, query_params: dict[str, Any]
    ):
        self.collection = collection
        self.query_params = query_params
        self.filter_query: dict[str, Any] = {}
        self.sort_query: list = [("createdAt", -1)]
        self.projection: dict[str, int] | None = None
        self.skip_count: int = 0
        self.limit_count: int = 100

    def filter(self) -> "APIFeatures":
        """Apply filtering to query."""
        query_obj = {**self.query_params}
        excluded_fields = [
            "page",
            "sort",
            "limit",
            "fields",
            "cursor",
            "direction",
            "sortField",
        ]

        for field in excluded_fields:
            query_obj.pop(field, None)

        # Advanced filtering: gte, gt, lte, lt
        query_str = str(query_obj)
        query_str = re.sub(r"\b(gte|gt|lte|lt)\b", r"$\1", query_str)

        # Convert operators
        for key, value in query_obj.items():
            if isinstance(value, str):
                if "gte:" in value:
                    self.filter_query[key] = {"$gte": value.replace("gte:", "")}
                elif "gt:" in value:
                    self.filter_query[key] = {"$gt": value.replace("gt:", "")}
                elif "lte:" in value:
                    self.filter_query[key] = {"$lte": value.replace("lte:", "")}
                elif "lt:" in value:
                    self.filter_query[key] = {"$lt": value.replace("lt:", "")}
                else:
                    self.filter_query[key] = value
            else:
                self.filter_query[key] = value

        return self

    def sort(self) -> "APIFeatures":
        """Apply sorting to query."""
        if self.query_params.get("sort"):
            sort_fields = self.query_params["sort"].split(",")
            self.sort_query = []
            for field in sort_fields:
                if field.startswith("-"):
                    self.sort_query.append((field[1:], -1))
                else:
                    self.sort_query.append((field, 1))

        return self

    def limit_fields(self) -> "APIFeatures":
        """Apply field limiting (projection)."""
        if self.query_params.get("fields"):
            fields = self.query_params["fields"].split(",")
            self.projection = {field: 1 for field in fields}
        else:
            self.projection = {"__v": 0}

        return self

    def paginate(self) -> "APIFeatures":
        """Apply offset-based pagination."""
        page = int(self.query_params.get("page", 1))
        limit = int(self.query_params.get("limit", 100))
        self.skip_count = (page - 1) * limit
        self.limit_count = limit

        return self

    def cursor_paginate(self) -> "APIFeatures":
        """Apply cursor-based pagination."""
        limit = int(self.query_params.get("limit", 10))
        cursor = self.query_params.get("cursor")
        direction = self.query_params.get("direction", "next").lower()
        sort_field = self.query_params.get("sortField", "_id")

        if cursor:
            if direction == "next":
                self.filter_query[sort_field] = {"$gt": cursor}
                self.sort_query = [(sort_field, 1)]
            else:
                self.filter_query[sort_field] = {"$lt": cursor}
                self.sort_query = [(sort_field, -1)]
        else:
            self.sort_query = [(sort_field, 1 if direction == "next" else -1)]

        self.limit_count = limit

        return self

    async def execute(self) -> list:
        """Execute the query and return results."""
        cursor = self.collection.find(self.filter_query, self.projection)

        if self.sort_query:
            cursor = cursor.sort(self.sort_query)

        if self.skip_count:
            cursor = cursor.skip(self.skip_count)

        if self.limit_count:
            cursor = cursor.limit(self.limit_count)

        return await cursor.to_list(length=None)
