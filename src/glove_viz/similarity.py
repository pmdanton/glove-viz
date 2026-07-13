import numpy as np
import pandas as pd


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bias_table(
    words: list[str],
    vectors: np.ndarray,
    anchor_man: np.ndarray,
    anchor_woman: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for word, vec in zip(words, vectors):
        s_man = cosine_similarity(vec, anchor_man)
        s_woman = cosine_similarity(vec, anchor_woman)
        rows.append({
            "word": word,
            "sim_man": round(s_man, 4),
            "sim_woman": round(s_woman, 4),
            "difference": round(s_man - s_woman, 4),
        })
    df = pd.DataFrame(rows)
    df["abs_diff"] = df["difference"].abs()
    df = df.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"])
    return df.reset_index(drop=True)


def compute_bias_scores(
    words: list[str],
    vectors: np.ndarray,
    anchor_man: np.ndarray,
    anchor_woman: np.ndarray,
) -> np.ndarray:
    scores = np.array([
        cosine_similarity(v, anchor_man) - cosine_similarity(v, anchor_woman)
        for v in vectors
    ])
    return scores
