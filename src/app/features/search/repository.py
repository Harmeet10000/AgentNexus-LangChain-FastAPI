
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


class SearchRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.searches

    async def list(
        self,
        *,
        user_id: ObjectId,
        page: int,
        limit: int,
        sort: str,
        mode: str | None,
    ):
        filter_q: dict = {"user_id": user_id}
        if mode:
            filter_q["transport_mode"] = mode

        total = await self.collection.count_documents(filter_q)
        skip = (page - 1) * limit

        sort_field = sort.lstrip("-")
        direction = -1 if sort.startswith("-") else 1

        cursor = (
            self.collection.find(filter_q)
            .sort(sort_field, direction)
            .skip(skip)
            .limit(limit)
        )

        data = await cursor.to_list(length=limit)
        return data, total

    async def get(self, *, search_id: ObjectId, user_id: ObjectId):
        return await self.collection.find_one({"_id": search_id, "user_id": user_id})

    async def delete(self, *, search_id: ObjectId, user_id: ObjectId):
        result = await self.collection.delete_one(
            {"_id": search_id, "user_id": user_id}
        )
        return result.deleted_count == 1

    async def stats(self, *, user_id: ObjectId):
        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "total_searches": {"$sum": 1},
                    "avg_cargo_weight": {"$avg": "$cargo_weight_kg"},
                    "total_co2_saved": {
                        "$sum": {
                            "$subtract": [
                                "$shortest_route.co2_emissions_kg",
                                "$efficient_route.co2_emissions_kg",
                            ]
                        }
                    },
                }
            },
        ]

        result = await self.collection.aggregate(pipeline).to_list(1)
        return result[0] if result else None
