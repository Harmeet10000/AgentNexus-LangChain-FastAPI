# violation: SQLAlchemy handler returns Failure without rollback
class RollbackViolation:
    pass
from sqlalchemy.exc import SQLAlchemyError
from returns.result import Failure
async def foo(session):
    try:
        await session.execute("SELECT 1")
    except SQLAlchemyError as exc:
        return Failure(None)
