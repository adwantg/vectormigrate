from __future__ import annotations

import json

import numpy as np

from vectormigrate import LowRankAffineAdapter, OrthogonalProcrustesAdapter, ResidualMLPAdapter


def main() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    target = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.55]])

    procrustes = OrthogonalProcrustesAdapter().fit(source, target)
    affine = LowRankAffineAdapter(rank=2).fit(source, target)
    mlp = ResidualMLPAdapter(hidden_dim=6, epochs=200, seed=3).fit(source, target)

    print(
        json.dumps(
            {
                "procrustes_mean_cosine": procrustes.mean_cosine_similarity(source, target),
                "affine_mean_cosine": affine.mean_cosine_similarity(source, target),
                "mlp_mean_cosine": mlp.mean_cosine_similarity(source, target),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
