from app.database.seeders.users import UserSeeder

from app.utils.logger import logger


async def run_all_seeders():
    """Run all seeders in order."""
    seeders = [
        UserSeeder(),
        # Add more seeders here
    ]

    logger.info("Starting database seeding")
    for seeder in seeders:
        await seeder.run()
    logger.info("Database seeding completed")
