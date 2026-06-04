from fastapi import APIRouter

from apps.api.routes import datasets, health, models, profiles, runs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(profiles.router)
api_router.include_router(datasets.router)
api_router.include_router(models.router)
api_router.include_router(runs.router)
