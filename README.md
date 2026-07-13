# 🧠 GloVe Gender Bias Explorer

Interactive Streamlit app visualising how word embeddings encode gender stereotypes.

## Features

- **4 projection methods:** Gender Axis, PCA, t-SNE, UMAP
- **Pre-built word groups:** Jobs, Family, Royalty, Gendered Pairs, Adjectives, Socioeconomic
- **Bias analysis:** Bar chart + table showing cosine similarity to male/female anchors
- **Word analogy explorer:** `king - man + woman ≈ queen`
- **~170 curated words** — no multi-GB download needed (71 KB subset, 50d)

## Run locally

```bash
cd glove-viz
uv run streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select this repo, set main file to `app.py`
4. Deploy — starts instantly, no download required

## Data

Uses a curated subset of [GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/) embeddings (Pennington, Socher & Manning 2014). Only ~170 relevant words are included (~433 KB) — jobs, gendered terms, socioeconomic adjectives, and family/royalty nouns.

To regenerate the subset from the full GloVe file:

```bash
uv run python extract_subset.py
```
