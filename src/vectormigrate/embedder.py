from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vectormigrate.models import EmbeddingABI


class Embedder(Protocol):
    abi: EmbeddingABI

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


def random_orthogonal_matrix(dimensions: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dimensions, dimensions))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return np.asarray(q * signs, dtype=np.float64)


class DeterministicHashEmbedder:
    """Deterministic embedder for tests, demos, and offline prototyping."""

    def __init__(
        self,
        abi: EmbeddingABI,
        semantic_salt: str = "vectormigrate",
        rotation_seed: int | None = None,
    ) -> None:
        self.abi = abi
        self.semantic_salt = semantic_salt
        if rotation_seed is None:
            self.rotation = np.eye(abi.dimensions, dtype=np.float64)
        else:
            self.rotation = random_orthogonal_matrix(abi.dimensions, rotation_seed)

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        rows = [self._embed_text(text) for text in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.abi.dimensions), dtype=np.float64)

    def _embed_text(self, text: str) -> NDArray[np.float64]:
        vector = np.zeros(self.abi.dimensions, dtype=np.float64)
        tokens = self._features(text)
        if not tokens:
            tokens = ["<empty>"]

        for feature in tokens:
            digest = hashlib.sha256(f"{self.semantic_salt}:{feature}".encode()).digest()
            for offset in range(0, 24, 8):
                chunk = digest[offset : offset + 8]
                index = int.from_bytes(chunk[:4], "big") % self.abi.dimensions
                sign = 1.0 if chunk[4] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[5] / 255.0)
                vector[index] += sign * magnitude

        projected = vector @ self.rotation
        if self.abi.normalization in {"l2", "unit"}:
            norm = float(np.linalg.norm(projected))
            if norm > 0:
                projected = projected / norm
        return projected

    def _features(self, text: str) -> list[str]:
        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9]+", normalized)
        features = list(tokens)
        features.extend(f"{left}::{right}" for left, right in pairwise(tokens))
        compact = normalized.replace(" ", "")
        features.extend(compact[index : index + 3] for index in range(max(len(compact) - 2, 0)))
        return features
