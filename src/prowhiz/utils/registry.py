"""Simple registry pattern for models and losses.

Allows instantiation by name from config without hard-coded if-else chains.

Usage:
    registry = Registry("models")

    @registry.register("egnn")
    class MyModel(nn.Module): ...

    model_cls = registry["egnn"]
    model = model_cls(**kwargs)
"""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Name → class mapping with decorator-based registration.

    Args:
        name: Human-readable name for this registry (used in error messages).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[Any]] = {}

    def register(self, key: str) -> Any:
        """Decorator that registers a class under `key`."""

        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry:
                raise KeyError(f"Registry '{self._name}': key '{key}' already registered")
            self._registry[key] = cls
            return cls

        return decorator

    def __getitem__(self, key: str) -> type[Any]:
        if key not in self._registry:
            raise KeyError(
                f"Registry '{self._name}': unknown key '{key}'. "
                f"Available: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self) -> list[str]:
        return sorted(self._registry.keys())
