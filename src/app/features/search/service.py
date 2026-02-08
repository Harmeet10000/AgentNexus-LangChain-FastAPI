from math import ceil

from bson import ObjectId

from app.utils.logger import logger


class SearchService:
    def __init__(self, repo, redis):
        self.repo = repo
        self.redis = redis

    def _serialize_search(self, doc):
        """Convert MongoDB document to serializable dict"""
        if not doc:
            return None
        return {
            "id": str(doc["_id"]),
            "user_id": str(doc["user_id"]),
            "origin": doc["origin"],
            "destination": doc["destination"],
            "cargo_weight_kg": doc["cargo_weight_kg"],
            "transport_mode": doc["transport_mode"],
            "shortest_route": doc["shortest_route"],
            "efficient_route": doc["efficient_route"],
            "metadata": doc.get("metadata", {}),
            "created_at": doc["created_at"],
        }

    async def list_searches(self, *, user_id, page, limit, sort, mode):
        # logger.info(
        #     "Listing searches",
        #     user_id=user_id,
        #     page=page,
        #     limit=limit,
        #     sort=sort,
        #     mode=mode,
        # )
        data, total = await self.repo.list(
            user_id=user_id,
            page=page,
            limit=limit,
            sort=sort,
            mode=mode,
        )
        # logger.info(f"Retrieved {len(data)} searches", data=data)

        total_pages = ceil(total / limit) if total else 0

        return {
            "data": [self._serialize_search(doc) for doc in data],
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
            },
        }

    async def get_search(self, *, search_id, user_id):
        doc = await self.repo.get(
            search_id=ObjectId(search_id),
            user_id=user_id,
        )
        return self._serialize_search(doc)

    async def delete_search(self, *, search_id, user_id):
        return await self.repo.delete(
            search_id=ObjectId(search_id),
            user_id=user_id,
        )

    async def get_stats(self, *, user_id):
        stats = await self.repo.stats(user_id=user_id)
        return {
            "total_searches": stats["total_searches"] if stats else 0,
            "total_co2_saved": stats["total_co2_saved"] if stats else 0.0,
            "avg_cargo_weight": stats["avg_cargo_weight"] if stats else 0.0,
        }
