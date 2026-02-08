from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis

from app.features.auth.model import User


class UserRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def get_by_email(self, email: str) -> User | None:
        return await User.find_one({"email": email})

    async def get_by_id(self, user_id: str) -> User | None:
        return await User.get(user_id)

    async def create(self, user: User) -> User:
        await user.insert()
        return user


class RefreshTokenRepository:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def store(self, jti: str, user_id: str, ttl_seconds: int):
        await self.redis.setex(f"refresh_token:{jti}", ttl_seconds, user_id)

    async def exists(self, jti: str) -> bool:
        return await self.redis.exists(f"refresh_token:{jti}") == 1

    async def revoke(self, jti: str):
        await self.redis.delete(f"refresh_token:{jti}")
