"""API 라우터 묶음.

**URL에 버전 접두사를 붙이지 않는다.** API 명세서의 경로가 `/analysis`,
`/animals`, `/spaces` 이므로 그대로 노출한다. 패키지 이름의 `v1` 은 소스 구조상의
구분이며, 버전 분기가 실제로 필요해지면 그때 접두사를 도입한다.
"""

from fastapi import APIRouter

from app.api.v1 import analysis, meta

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(meta.router)
api_router.include_router(analysis.router)
