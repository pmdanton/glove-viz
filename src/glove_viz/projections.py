import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def project_pca(vectors: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


def project_tsne(vectors: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    n_samples = len(vectors)
    effective_perplexity = min(perplexity, max(2, (n_samples - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=float(effective_perplexity),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(vectors)


def project_umap(vectors: np.ndarray, n_neighbors: int = 15) -> np.ndarray | None:
    try:
        import umap
    except ImportError:
        return None
    effective_neighbors = min(n_neighbors, max(2, len(vectors) - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=effective_neighbors, random_state=42)
    return reducer.fit_transform(vectors)


def project_gender_axis(
    vectors: np.ndarray,
    anchor_man: np.ndarray,
    anchor_woman: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_gender = anchor_man - anchor_woman
    v_gender = v_gender / np.linalg.norm(v_gender)

    rng = np.random.RandomState(42)
    rand = rng.randn(len(v_gender))
    v_orth = rand - np.dot(rand, v_gender) * v_gender
    v_orth = v_orth / np.linalg.norm(v_orth)

    x = vectors @ v_gender
    y = vectors @ v_orth
    coords = np.column_stack([x, y])
    return coords, v_gender, v_orth
