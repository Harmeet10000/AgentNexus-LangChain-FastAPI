from fastapi import APIRouter, Response
router = APIRouter()
@router.get("/x")
async def get_x(response: Response):
    raise ForbiddenRouterError()
