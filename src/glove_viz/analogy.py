"""Word analogy solver: a is to b as c is to ?"""

from __future__ import annotations

import numpy as np


def solve_analogy(
    a: str,
    b: str,
    c: str,
    glove: dict[str, np.ndarray],
    top_k: int = 5,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Solve: a is to b as c is to ?
    
    Computes: result_vec = vec(b) - vec(a) + vec(c)
    Returns top_k nearest neighbours by cosine similarity, excluding input words.
    """
    exclude = exclude or {a.lower(), b.lower(), c.lower()}

    vec_a = glove.get(a.lower())
    vec_b = glove.get(b.lower())
    vec_c = glove.get(c.lower())

    if vec_a is None or vec_b is None or vec_c is None:
        missing = [w for w, v in [(a, vec_a), (b, vec_b), (c, vec_c)] if v is None]
        raise ValueError(f"Words not in vocabulary: {missing}")

    target = vec_b - vec_a + vec_c
    target_norm = target / np.linalg.norm(target)

    # Score all words (brute force — GloVe is ~400k words, still fast)
    results: list[tuple[str, float]] = []
    for word, vec in glove.items():
        if word in exclude:
            continue
        sim = float(np.dot(target_norm, vec / np.linalg.norm(vec)))
        results.append((word, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
