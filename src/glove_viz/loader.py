"""Load the curated GloVe subset (~170 words, 50d).

No download needed — glove_subset.txt is committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SUBSET_FILE = DATA_DIR / "glove_subset.txt"


@st.cache_data(show_spinner=False)
def load_glove() -> dict[str, np.ndarray]:
    """Load the curated GloVe subset into a dict. Cached across reruns."""
    if not SUBSET_FILE.exists():
        st.error(
            f"Subset file not found: {SUBSET_FILE}\n"
            "Run `python extract_subset.py` to generate it from the full GloVe file."
        )
        st.stop()

    vectors: dict[str, np.ndarray] = {}
    with open(SUBSET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors


def get_vectors(words: list[str], glove: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return (valid_words, matrix) for words present in the vocabulary.

    Skips words not found.
    """
    valid = []
    vecs = []
    for w in words:
        w_lower = w.strip().lower()
        if w_lower in glove:
            valid.append(w_lower)
            vecs.append(glove[w_lower])
    if not vecs:
        return [], np.empty((0, 300))
    return valid, np.vstack(vecs)
