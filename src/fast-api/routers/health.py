"""Health and model discovery endpoints."""

from __future__ import annotations

from typing import List
from fastapi import APIRouter

from core.registry import model_registry
from core.schemas import HealthResponse, ModelMetadata

router = APIRouter(tags=["System & Discovery"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Check backend operational health and list available/loaded models."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_models=model_registry.list_available_models(),
        loaded_models=model_registry.list_loaded_models(),
    )


@router.get(
    "/api/v1/models",
    response_model=List[ModelMetadata],
    summary="List all registered vision models",
)
async def list_models() -> List[ModelMetadata]:
    """Retrieve metadata, class taxonomies, and status for all registered models."""
    return model_registry.get_all_metadata()
