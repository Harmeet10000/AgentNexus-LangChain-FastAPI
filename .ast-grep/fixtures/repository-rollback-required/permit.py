# permit: with rollback and read-only
from sqlalchemy.exc import SQLAlchemyError
from returns.result import Failure
async def foo(session):
    try:
        await session.execute("SELECT 1")
    except SQLAlchemyError as exc:
        await session.rollback()
        return Failure(None)
