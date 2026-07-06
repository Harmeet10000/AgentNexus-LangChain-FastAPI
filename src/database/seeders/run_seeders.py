from app.utils.logger import logger


async def run_all_seeders() -> None:
    """Run all seeders in order."""
    logger.info("Starting database seeding")
    # Add concrete seeders here as they are implemented.
    logger.info("Database seeding completed")
