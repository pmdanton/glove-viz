# GloVe Embedding Visualiser — Implementation Plan

## Goal
Streamlit app to visualise GloVe word embeddings in 2D, with focus on illustrating gender bias in word embeddings (e.g., "surgeon" closer to "man" than "woman").

## File Structure
```
glove-viz/
├── app.py                  # Streamlit entry point
├── src/
│   └── glove_viz/
│       ├── __init__.py
│       ├── loader.py       # GloVe download + caching
│       ├── projections.py  # PCA, t-SNE, UMAP, gender-axis
│       ├── similarity.py   # Cosine similarity, bias analysis
│       ├── analogy.py      # Word analogy solver
│       └── presets.py      # Curated word lists
├── data/                   # GloVe files (gitignored)
├── pyproject.toml
└── PLAN.md
```

## Module Breakdown

### 1. `loader.py` — GloVe Data Loading
- URL: `https://nlp.stanford.edu/data/glove.6B.zip`
- Extract `glove.6B.300d.txt` to `data/`
- Parse into `dict[str, np.ndarray]` (word → 300d vector)
- `@st.cache_data` decorator for memoisation
- Show spinner during load (~822MB zip, ~300MB extracted)

### 2. `projections.py` — 2D Projections
Four methods, all take `words: list[str], vectors: np.ndarray (n, 300)` → `np.ndarray (n, 2)`

- **PCA**: `sklearn.decomposition.PCA(n_components=2)`. Fast, linear, preserves global structure.
- **t-SNE**: `sklearn.manifold.TSNE(n_components=2, perplexity=...)`. Non-linear, reveals clusters. Expose perplexity slider (5–50, default 30).
- **UMAP**: Try `import umap`, fallback gracefully if not installed. `umap.UMAP(n_components=2)`.
- **Gender Axis**: 
  - `v_gender = vec("man") - vec("woman")`, normalise to unit vector
  - `v_orth = random_orthogonal(v_gender)` (via QR or Gram-Schmidt)
  - x-axis = projection onto v_gender, y-axis = projection onto v_orth
  - Draw gender axis line on plot, label "← woman" and "man →"

### 3. `similarity.py` — Bias Analysis
- `cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))`
- `bias_table(words, vec("man"), vec("woman"))` → DataFrame with columns: word, sim_to_man, sim_to_woman, difference
- Bar chart: grouped bars (man similarity blue, woman similarity red)
- Highlight words where `|sim_man - sim_woman| > 0.05`

### 4. `analogy.py` — Word Analogies
- `solve_analogy(a, b, c, vectors)` → `b - a + c`, find top-k nearest neighbours by cosine similarity
- Exclude input words from results
- Pre-fill: king - man + woman

### 5. `presets.py` — Curated Word Lists
```python
PRESETS = {
    "Jobs": ["surgeon", "doctor", "nurse", "teacher", "engineer", "programmer", 
             "secretary", "soldier", "pilot", "ceo", "receptionist", "mechanic",
             "librarian", "chef", "dentist", "lawyer", "scientist"],
    "Family": ["mother", "father", "sister", "brother", "wife", "husband",
               "grandmother", "grandfather", "daughter", "son", "aunt", "uncle"],
    "Royalty": ["king", "queen", "prince", "princess", "monarch", "ruler",
                "throne", "crown", "noble", "lord", "lady", "duke", "duchess"],
    "Gendered pairs": ["man", "woman", "male", "female", "masculine", "feminine",
                       "boy", "girl", "gentleman", "lady", "husband", "wife"],
}
```

## Streamlit Layout

```
┌─────────────────────────────────────────────────┐
│  GloVe Gender Bias Explorer                     │
├────────────┬────────────────────────────────────┤
│ SIDEBAR    │  MAIN AREA                         │
│            │                                     │
│ Projection │  [Scatter Plot — full width]        │
│ method ○   │  Colour: sim to man (blue)          │
│            │         sim to woman (red)           │
│ Perplexity │                                     │
│ slider     │  ─────────────────────────────────  │
│ (t-SNE)    │                                     │
│            │  [Bias Analysis]                    │
│ Preset     │  Bar chart: cos sim man vs woman    │
│ selector   │  Table with exact values            │
│            │                                     │
│ Custom     │  ─────────────────────────────────  │
│ words      │                                     │
│ (text area)│  [Word Analogy Explorer]            │
│            │  king - man + woman = ?              │
│ Gender     │  Top-5 nearest neighbours           │
│ anchors    │                                     │
│ (man/woman)│                                     │
└────────────┴────────────────────────────────────┘
```

## Key Implementation Details

### Colour Mapping
For each word, compute `bias = cos_sim(w, man) - cos_sim(w, woman)`, normalise to [-1, 1].
Map to colour: -1 = red (feminine), +1 = blue (masculine). Use matplotlib `RdBu` or custom.

### Gender Axis Orthogonal
```python
v_gender = vec_man - vec_woman
v_gender /= np.linalg.norm(v_gender)
# Gram-Schmidt: pick a random vector, subtract its projection onto v_gender
rand = np.random.randn(300)
v_orth = rand - np.dot(rand, v_gender) * v_gender
v_orth /= np.linalg.norm(v_orth)
```

### GloVe Download
- Use `urllib.request.urlretrieve` with progress callback
- Use `zipfile.ZipFile` to extract just `glove.6B.300d.txt`
- Cache path: `data/glove.6B.300d.txt`

## Dependencies
Already installed: streamlit, numpy, scikit-learn
Optional: umap-learn (for UMAP projection)
