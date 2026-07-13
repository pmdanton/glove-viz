import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from src.glove_viz.loader import load_glove, get_vectors
from src.glove_viz.projections import project_pca, project_tsne, project_gender_axis
from src.glove_viz.similarity import bias_table, compute_bias_scores, cosine_similarity
from src.glove_viz.analogy import solve_analogy
from src.glove_viz.presets import PRESETS
import numpy as np

print("Loading GloVe embeddings...")
glove = load_glove()
print(f"Loaded {len(glove)} word vectors")

for w in ["man", "woman", "king", "queen", "surgeon", "doctor", "nurse", "engineer"]:
    if w in glove:
        print(f"  '{w}': vector shape {glove[w].shape}")
    else:
        print(f"  '{w}': MISSING")

words = ["man", "woman", "king", "queen", "surgeon", "doctor", "nurse", "engineer", "programmer"]
valid_words, vectors = get_vectors(words, glove)
print(f"\nValid words: {valid_words}")

# PCA
coords = project_pca(vectors)
print(f"PCA projection shape: {coords.shape}")

# t-SNE
coords = project_tsne(vectors, perplexity=5)
print(f"t-SNE projection shape: {coords.shape}")

# Gender Axis
coords, v_gender, v_orth = project_gender_axis(vectors, glove["man"], glove["woman"])
print(f"Gender Axis projection shape: {coords.shape}")
print(f"Gender direction shape: {v_gender.shape}")

# Bias table
df = bias_table(valid_words, vectors, glove["man"], glove["woman"])
print(f"\nBias table:\n{df}")

# compute_bias_scores
scores = compute_bias_scores(valid_words, vectors, glove["man"], glove["woman"])
print(f"\nBias scores: {scores}")

# Analogy
results = solve_analogy("king", "man", "woman", glove, top_k=5)
print(f"\nAnalogy: king - man + woman =")
for w, sim in results:
    print(f"  {w}: {sim:.4f}")

# Presets
print(f"\nPresets: {list(PRESETS.keys())}")

# Cosine similarity direct test
s = cosine_similarity(glove["king"], glove["queen"])
print(f"\ncos(king, queen) = {s:.4f}")

print("\nAll tests passed!")
