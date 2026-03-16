from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class CompatibilityAdapter(Protocol):
    def transform(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def confidence(self, vectors: NDArray[np.float64]) -> float: ...


class OrthogonalProcrustesAdapter:
    """Maps vectors from a source embedding space into a target space."""

    def __init__(self, center: bool = True) -> None:
        self.center = center
        self._source_mean: NDArray[np.float64] | None = None
        self._target_mean: NDArray[np.float64] | None = None
        self._rotation: NDArray[np.float64] | None = None
        self._source_radius: float | None = None

    def fit(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> OrthogonalProcrustesAdapter:
        if source_vectors.shape != target_vectors.shape:
            raise ValueError("source and target vectors must have the same shape")
        if source_vectors.ndim != 2:
            raise ValueError("source and target vectors must be 2-dimensional")
        if source_vectors.shape[0] == 0:
            raise ValueError("at least one paired vector is required")

        source = np.asarray(source_vectors, dtype=np.float64)
        target = np.asarray(target_vectors, dtype=np.float64)
        if self.center:
            source_mean = source.mean(axis=0)
            target_mean = target.mean(axis=0)
        else:
            source_mean = np.zeros(source.shape[1], dtype=np.float64)
            target_mean = np.zeros(target.shape[1], dtype=np.float64)

        self._source_mean = source_mean
        self._target_mean = target_mean
        centered_source_norm = np.linalg.norm(source - source_mean, axis=1)
        self._source_radius = float(
            centered_source_norm.mean() + centered_source_norm.std() + 1e-12
        )
        centered_source = source - source_mean
        centered_target = target - target_mean
        u, _, vt = np.linalg.svd(centered_source.T @ centered_target, full_matrices=False)
        self._rotation = u @ vt
        return self

    def transform(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._rotation is None or self._source_mean is None or self._target_mean is None:
            raise RuntimeError("adapter must be fit before use")
        rotation = self._rotation
        source_mean = self._source_mean
        target_mean = self._target_mean
        matrix = np.asarray(vectors, dtype=np.float64)
        return (matrix - source_mean) @ rotation + target_mean

    def confidence(self, vectors: NDArray[np.float64]) -> float:
        if self._source_mean is None or self._source_radius is None:
            raise RuntimeError("adapter must be fit before use")
        distances = np.linalg.norm(vectors - self._source_mean, axis=1)
        raw = 1.0 - (distances / self._source_radius)
        return float(np.clip(raw.mean(), 0.0, 1.0))

    def mean_cosine_similarity(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> float:
        transformed = self.transform(source_vectors)
        transformed_norm = np.linalg.norm(transformed, axis=1)
        target_norm = np.linalg.norm(target_vectors, axis=1)
        valid = (transformed_norm > 0) & (target_norm > 0)
        if not np.any(valid):
            return 0.0
        cosine = np.sum(transformed[valid] * target_vectors[valid], axis=1) / (
            transformed_norm[valid] * target_norm[valid]
        )
        return float(cosine.mean())


class LowRankAffineAdapter:
    """Affine compatibility adapter with optional low-rank truncation."""

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self._weight: NDArray[np.float64] | None = None
        self._bias: NDArray[np.float64] | None = None
        self._source_mean: NDArray[np.float64] | None = None
        self._source_radius: float | None = None

    def fit(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> LowRankAffineAdapter:
        if source_vectors.shape != target_vectors.shape:
            raise ValueError("source and target vectors must have the same shape")
        if source_vectors.ndim != 2:
            raise ValueError("source and target vectors must be 2-dimensional")
        if source_vectors.shape[0] == 0:
            raise ValueError("at least one paired vector is required")

        source = np.asarray(source_vectors, dtype=np.float64)
        target = np.asarray(target_vectors, dtype=np.float64)
        source_augmented = np.hstack([source, np.ones((source.shape[0], 1), dtype=np.float64)])
        solution, _, _, _ = np.linalg.lstsq(source_augmented, target, rcond=None)
        weight = solution[:-1, :]
        bias = solution[-1, :]
        if self.rank is not None and self.rank < min(weight.shape):
            u, singular_values, vt = np.linalg.svd(weight, full_matrices=False)
            truncated = (u[:, : self.rank] * singular_values[: self.rank]) @ vt[: self.rank, :]
            weight = np.asarray(truncated, dtype=np.float64)

        self._weight = np.asarray(weight, dtype=np.float64)
        self._bias = np.asarray(bias, dtype=np.float64)
        source_mean = source.mean(axis=0)
        self._source_mean = source_mean
        centered_source_norm = np.linalg.norm(source - source_mean, axis=1)
        self._source_radius = float(
            centered_source_norm.mean() + centered_source_norm.std() + 1e-12
        )
        return self

    def transform(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._weight is None or self._bias is None:
            raise RuntimeError("adapter must be fit before use")
        matrix = np.asarray(vectors, dtype=np.float64)
        return matrix @ self._weight + self._bias

    def confidence(self, vectors: NDArray[np.float64]) -> float:
        if self._source_mean is None or self._source_radius is None:
            raise RuntimeError("adapter must be fit before use")
        distances = np.linalg.norm(vectors - self._source_mean, axis=1)
        raw = 1.0 - (distances / self._source_radius)
        return float(np.clip(raw.mean(), 0.0, 1.0))

    def mean_cosine_similarity(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> float:
        transformed = self.transform(source_vectors)
        transformed_norm = np.linalg.norm(transformed, axis=1)
        target_norm = np.linalg.norm(target_vectors, axis=1)
        valid = (transformed_norm > 0) & (target_norm > 0)
        if not np.any(valid):
            return 0.0
        cosine = np.sum(transformed[valid] * target_vectors[valid], axis=1) / (
            transformed_norm[valid] * target_norm[valid]
        )
        return float(cosine.mean())


class ResidualMLPAdapter:
    """Small numpy MLP adapter with a residual connection."""

    def __init__(
        self,
        hidden_dim: int = 16,
        learning_rate: float = 0.05,
        epochs: int = 200,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self._w1: NDArray[np.float64] | None = None
        self._b1: NDArray[np.float64] | None = None
        self._w2: NDArray[np.float64] | None = None
        self._b2: NDArray[np.float64] | None = None
        self._source_mean: NDArray[np.float64] | None = None
        self._source_radius: float | None = None

    def fit(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> ResidualMLPAdapter:
        if source_vectors.shape != target_vectors.shape:
            raise ValueError("source and target vectors must have the same shape")
        if source_vectors.ndim != 2:
            raise ValueError("source and target vectors must be 2-dimensional")
        if source_vectors.shape[0] == 0:
            raise ValueError("at least one paired vector is required")

        source = np.asarray(source_vectors, dtype=np.float64)
        target = np.asarray(target_vectors, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        input_dim = source.shape[1]
        self._w1 = rng.normal(scale=0.1, size=(input_dim, self.hidden_dim))
        self._b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self._w2 = rng.normal(scale=0.1, size=(self.hidden_dim, input_dim))
        self._b2 = np.zeros(input_dim, dtype=np.float64)
        source_mean = source.mean(axis=0)
        self._source_mean = source_mean
        centered_source_norm = np.linalg.norm(source - source_mean, axis=1)
        self._source_radius = float(
            centered_source_norm.mean() + centered_source_norm.std() + 1e-12
        )

        for _ in range(self.epochs):
            hidden_linear = source @ self._w1 + self._b1
            hidden = np.tanh(hidden_linear)
            output = source + hidden @ self._w2 + self._b2
            error = output - target
            grad_output = (2.0 / source.shape[0]) * error
            grad_w2 = hidden.T @ grad_output
            grad_b2 = grad_output.sum(axis=0)
            grad_hidden = (grad_output @ self._w2.T) * (1.0 - np.square(hidden))
            grad_w1 = source.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            self._w2 -= self.learning_rate * grad_w2
            self._b2 -= self.learning_rate * grad_b2
            self._w1 -= self.learning_rate * grad_w1
            self._b1 -= self.learning_rate * grad_b1
        return self

    def transform(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._w1 is None or self._b1 is None or self._w2 is None or self._b2 is None:
            raise RuntimeError("adapter must be fit before use")
        matrix = np.asarray(vectors, dtype=np.float64)
        hidden = np.tanh(matrix @ self._w1 + self._b1)
        return np.asarray(matrix + hidden @ self._w2 + self._b2, dtype=np.float64)

    def confidence(self, vectors: NDArray[np.float64]) -> float:
        if self._source_mean is None or self._source_radius is None:
            raise RuntimeError("adapter must be fit before use")
        distances = np.linalg.norm(vectors - self._source_mean, axis=1)
        raw = 1.0 - (distances / self._source_radius)
        return float(np.clip(raw.mean(), 0.0, 1.0))

    def mean_cosine_similarity(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> float:
        transformed = self.transform(source_vectors)
        transformed_norm = np.linalg.norm(transformed, axis=1)
        target_norm = np.linalg.norm(target_vectors, axis=1)
        valid = (transformed_norm > 0) & (target_norm > 0)
        if not np.any(valid):
            return 0.0
        cosine = np.sum(transformed[valid] * target_vectors[valid], axis=1) / (
            transformed_norm[valid] * target_norm[valid]
        )
        return float(cosine.mean())
