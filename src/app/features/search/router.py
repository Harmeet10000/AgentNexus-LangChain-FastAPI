from fastapi import APIRouter, Depends, HTTPException, Query

from app.features.auth.dependency import get_current_user
from app.features.search.dependency import get_search_service

router = APIRouter(prefix="/api/v1/searches", tags=["Searches"])


@router.get("")
async def list_searches(
    page: int = Query(1, ge=1),
    limit: int = Query(20, le=100),
    sort: str = "-created_at",
    mode: str | None = None,
    user=Depends(get_current_user),
    service=Depends(get_search_service),
):
    return await service.list_searches(
        user_id=user.id,
        page=page,
        limit=limit,
        sort=sort,
        mode=mode,
    )


@router.get("/{search_id}")
async def get_search(
    search_id: str,
    user=Depends(get_current_user),
    service=Depends(get_search_service),
):
    result = await service.get_search(
        search_id=search_id,
        user_id=user.id,
    )
    if not result:
        raise HTTPException(404, "Search not found")
    return result


@router.delete("/{search_id}", status_code=204)
async def delete_search(
    search_id: str,
    user=Depends(get_current_user),
    service=Depends(get_search_service),
):
    deleted = await service.delete_search(
        search_id=search_id,
        user_id=user.id,
    )
    if not deleted:
        raise HTTPException(404, "Search not found")


@router.get("/stats")
async def search_stats(
    user=Depends(get_current_user),
    service=Depends(get_search_service),
):
    return await service.get_stats(user_id=user.id)
