from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendCapabilities:
    supports_alias_swap: bool
    supports_reindex: bool
    supports_named_vectors: bool = False
    supports_server_side_rrf: bool = False


@dataclass(frozen=True)
class BackendOperation:
    name: str
    method: str
    path: str
    body: Mapping[str, Any] = field(default_factory=dict)
