from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any


@dataclass(frozen=True)
class BackendPlugin:
    name: str
    factory: Callable[..., Any]
    description: str = ""


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, BackendPlugin] = {}

    def register(self, plugin: BackendPlugin) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> BackendPlugin:
        return self._plugins[name]

    def list_plugins(self) -> list[BackendPlugin]:
        return [self._plugins[name] for name in sorted(self._plugins)]

    def load_entry_point_plugins(
        self,
        group: str = "vectormigrate.backends",
    ) -> list[BackendPlugin]:
        loaded: list[BackendPlugin] = []
        for entry_point in entry_points().select(group=group):
            plugin = self._load_entry_point(entry_point)
            self.register(plugin)
            loaded.append(plugin)
        return loaded

    @staticmethod
    def _load_entry_point(entry_point: EntryPoint) -> BackendPlugin:
        loaded = entry_point.load()
        description = getattr(loaded, "__doc__", "") or ""
        return BackendPlugin(name=entry_point.name, factory=loaded, description=description)
