"""Download, extract, and cache GloVe embeddings."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import streamlit as st

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.300d.txt"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a Streamlit progress bar."""
    progress_bar = st.progress(0, text="Downloading GloVe embeddings (~822 MB)…")

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(count * block_size / total_size, 1.0)
            progress_bar.progress(pct, text=f"Downloading GloVe embeddings… {pct:.0%}")

    urlretrieve(url, str(dest), reporthook=_reporthook)
    progress_bar.progress(1.0, text="Download complete!")
    progress_bar.empty()


def _ensure_glove_file() -> Path:
    """Make sure glove.6B.300d.txt exists locally; download + extract if needed."""
    glove_path = DATA_DIR / GLOVE_FILE
    if glove_path.exists():
        return glove_path

    zip_path = DATA_DIR / "glove.6B.zip"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        _download_with_progress(GLOVE_URL, zip_path)

    with st.spinner("Extracting GloVe vectors…"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only the 300d file
            with zf.open(GLOVE_FILE) as src, open(glove_path, "wb") as dst:
                dst.write(src.read())

    return glove_path


@st.cache_data(show_spinner=False)
def load_glove() -> dict[str, np.ndarray]:
    """Load GloVe 300d vectors into a dict. Cached across reruns."""
    glove_path = _ensure_glove_file()
    vectors: dict[str, np.ndarray] = {}

    with st.spinner("Loading GloVe vectors into memory…"):
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vec = np.array(parts[1:], dtype=np.float32)
                vectors[word] = vec

    return vectors


def get_vectors(words: list[str], glove: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return (valid_words, matrix) for words present in GloVe.
    
    Skips words not found in the vocabulary.
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
