from fastapi import APIRouter, Response

from app.utils.exceptions import NotFoundException, ValidationException, ForbiddenException

router = APIRouter()

# violation: router raises expected-failure exception instead of render_result
@router.get("/x")
async def get_x(response: Response):
    raise NotFoundException("not found")

@router.get("/y")
async def get_y(response: Response):
    raise ValidationException(detail="bad input")

@router.get("/z")
async def get_z(response: Response):
    raise ForbiddenException(detail="forbidden")
