# 🧠 GloVe Gender Bias Explorer

Interactive Streamlit app visualising how word embeddings encode gender stereotypes.

## Features

- **4 projection methods:** Gender Axis, PCA, t-SNE, UMAP
- **Pre-built word groups:** Jobs, Family, Royalty, Gendered Pairs, Adjectives
- **Bias analysis:** Bar chart + table showing cosine similarity to male/female anchors
- **Word analogy explorer:** `king - man + woman ≈ queen`

## Run locally

```bash
cd glove-viz
uv run streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select this repo, set main file to `app.py`
4. Deploy — GloVe vectors download automatically on first run (~822MB zip, one-time)

## Data

Uses [GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/) embeddings (Pennington, Socher & Manning 2014). Downloaded and cached automatically on first run.
