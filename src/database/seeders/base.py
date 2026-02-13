from abc import ABC, abstractmethod

from app.utils.logger import logger


class BaseSeeder(ABC):
    """Base class for all seeders."""

    @abstractmethod
    async def seed(self):
        """Implement seeding logic."""
        pass

    async def run(self):
        """Run the seeder with logging."""
        logger.info(f"Running seeder: {self.__class__.__name__}")
        try:
            await self.seed()
            logger.info(f"✓ Seeder completed: {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"✗ Seeder failed: {self.__class__.__name__}", exc_info=True)
            raise
