"""Hook for metrics this suite does not compute.

Word error rate and perimeter accuracy are the sort of measurement a host
may register. This package ships no provider for either one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict


class MetricReading(BaseModel):
    """One outside measurement. The map computes the Wilson interval itself."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    kind: Literal["proportion", "point"]
    sample_count: int | None = None
    successes: int | None = None
    source: Literal["live", "fixture", "smoke", "mixed"] | None = None
    detail: str = ""


class ExternalMetricProvider(ABC):
    """Interface a host implements and registers. Paths are declared up front."""

    name: str

    @abstractmethod
    def paths(self) -> frozenset[str]:
        """Subpaths under ``plugin.<name>.``, such as ``accuracy``."""

    @abstractmethod
    def read(self, run: dict, items: list[dict], path: str) -> MetricReading | None:
        """Return a reading, or None when this run has no such measurement."""


class MetricRegistry:
    """Process-local registry. Built-in suite paths do not go through it."""

    def __init__(self) -> None:
        self._providers: dict[str, ExternalMetricProvider] = {}

    def register(self, provider: ExternalMetricProvider) -> None:
        if not isinstance(provider, ExternalMetricProvider):
            raise TypeError("provider must implement ExternalMetricProvider")
        name = str(getattr(provider, "name", "") or "").strip()
        if not name.isidentifier():
            raise ValueError(f"provider name {name!r} must be an identifier")
        if name in self._providers:
            raise ValueError(f"metric provider {name!r} is already registered")
        if not provider.paths():
            raise ValueError(f"metric provider {name!r} must declare at least one path")
        self._providers[name] = provider

    def discard(self, name: str) -> None:
        self._providers.pop(name, None)

    def get(self, name: str) -> ExternalMetricProvider | None:
        return self._providers.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def __contains__(self, name: str) -> bool:
        return name in self._providers


default_registry = MetricRegistry()


def register_metric(
    provider: ExternalMetricProvider,
    registry: MetricRegistry | None = None,
) -> None:
    """Register ``provider`` on ``registry``, or on the process registry."""
    target = default_registry if registry is None else registry
    target.register(provider)
