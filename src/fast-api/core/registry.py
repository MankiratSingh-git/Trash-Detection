"""Central Model Registry for Sortify.

Allows dynamic registration, discovery, caching, and lifecycle management
for multiple vision models (YOLOv11, YOLOv8, classifiers, etc.).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Type
from .base_model import BaseVisionModel
from .schemas import ModelMetadata

logger = logging.getLogger("sortify.registry")


class ModelRegistry:
    """Singleton registry storing available vision models."""

    _instance: Optional[ModelRegistry] = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, BaseVisionModel] = {}
            cls._instance._factories: Dict[str, Callable[[], BaseVisionModel]] = {}
        return cls._instance

    def register(self, name: str, factory: Optional[Callable[[], BaseVisionModel]] = None) -> Callable:
        """Decorator or direct method to register a model factory."""
        def decorator(fn: Callable[[], BaseVisionModel]):
            self._factories[name.lower()] = fn
            logger.info("Registered model factory for: %s", name.lower())
            return fn

        if factory is not None:
            return decorator(factory)
        return decorator

    def register_instance(self, name: str, model_instance: BaseVisionModel) -> None:
        """Register an already instantiated model."""
        self._models[name.lower()] = model_instance
        logger.info("Registered model instance for: %s", name.lower())

    def get_model(self, name: str, load_if_needed: bool = True) -> BaseVisionModel:
        """Retrieve a registered model by name. Lazily loads if requested."""
        key = name.lower()

        # Check if already instantiated
        if key in self._models:
            model = self._models[key]
            if load_if_needed and not model.is_loaded:
                model.load()
            return model

        # Check if factory is registered
        if key in self._factories:
            model = self._factories[key]()
            self._models[key] = model
            if load_if_needed and not model.is_loaded:
                model.load()
            return model

        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Available models: {self.list_available_models()}"
        )

    def list_available_models(self) -> List[str]:
        """List names of all models registered either as instance or factory."""
        names = set(self._models.keys()).union(set(self._factories.keys()))
        return sorted(list(names))

    def list_loaded_models(self) -> List[str]:
        """List names of models currently loaded in memory."""
        return [k for k, v in self._models.items() if v.is_loaded]

    def get_all_metadata(self) -> List[ModelMetadata]:
        """Return metadata for all registered models."""
        metadata_list: List[ModelMetadata] = []
        for name in self.list_available_models():
            try:
                # Get without forcing heavy load if not loaded yet
                if name in self._models:
                    metadata_list.append(self._models[name].get_metadata())
                elif name in self._factories:
                    # Instantiate factory to read metadata
                    model = self.get_model(name, load_if_needed=False)
                    metadata_list.append(model.get_metadata())
            except Exception as e:
                logger.error("Failed to get metadata for model %s: %s", name, e)
        return metadata_list


# Global registry singleton
model_registry = ModelRegistry()
