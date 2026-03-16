from __future__ import annotations

import numpy as np
import pytest

from vectormigrate.compat import (
    LowRankAffineAdapter,
    OrthogonalProcrustesAdapter,
    ResidualMLPAdapter,
)
from vectormigrate.embedder import random_orthogonal_matrix


def test_procrustes_recovers_orthogonal_rotation() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 8))
    rotation = random_orthogonal_matrix(8, seed=11)
    target = source @ rotation

    adapter = OrthogonalProcrustesAdapter(center=False).fit(source, target)
    transformed = adapter.transform(source)

    assert np.allclose(transformed, target, atol=1e-8)
    assert adapter.mean_cosine_similarity(source, target) > 0.999999


def test_procrustes_rejects_shape_mismatch() -> None:
    adapter = OrthogonalProcrustesAdapter()

    with pytest.raises(ValueError):
        adapter.fit(np.ones((4, 3)), np.ones((5, 3)))


def test_low_rank_affine_learns_affine_mapping() -> None:
    rng = np.random.default_rng(21)
    source = rng.normal(size=(64, 6))
    weight = rng.normal(size=(6, 6))
    bias = rng.normal(size=(6,))
    target = source @ weight + bias

    adapter = LowRankAffineAdapter(rank=6).fit(source, target)
    transformed = adapter.transform(source)

    assert np.allclose(transformed, target, atol=1e-8)
    assert adapter.mean_cosine_similarity(source, target) > 0.999999
    assert 0.0 <= adapter.confidence(source[:2]) <= 1.0


def test_residual_mlp_adapter_learns_small_nonlinear_mapping() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(64, 5))
    target = source + 0.2 * np.tanh(source)

    adapter = ResidualMLPAdapter(hidden_dim=10, learning_rate=0.05, epochs=300, seed=5).fit(
        source,
        target,
    )
    transformed = adapter.transform(source)

    mse = np.mean(np.square(transformed - target))
    assert mse < 0.01
    assert adapter.mean_cosine_similarity(source, target) > 0.99
