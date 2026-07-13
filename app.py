"""GloVe Gender Bias Explorer — Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.glove_viz.loader import get_vectors, load_glove
from src.glove_viz.presets import GENDER_ANCHORS, PRESETS
from src.glove_viz.projections import project_gender_axis, project_pca, project_tsne, project_umap
from src.glove_viz.similarity import bias_table, compute_bias_scores, cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GloVe Gender Bias Explorer",
    page_icon="🧠",
    layout="wide",
)

# ── Title ────────────────────────────────────────────────────────────────────
st.title("🧠 GloVe Gender Bias Explorer")
st.markdown(
    "Visualise how word embeddings encode gender stereotypes — "
    "neutral job nouns like *surgeon* and *soldier* sit closer to *man* than *woman* in vector space."
)

# ── Load GloVe ───────────────────────────────────────────────────────────────
glove = load_glove()
dim = len(next(iter(glove.values())))
st.caption(f"Loaded **{len(glove):,}** curated word vectors (GloVe 6B, {dim}d subset)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Projection method
    projection = st.selectbox(
        "Projection method",
        ["Gender Axis", "PCA", "t-SNE", "UMAP"],
        index=0,
        help="'Gender Axis' projects onto the man–woman direction — best for showing bias.",
    )

    # t-SNE perplexity
    perplexity = 30.0
    if projection == "t-SNE":
        perplexity = st.slider("t-SNE perplexity", 5, 50, 30)

    # UMAP neighbours
    n_neighbors = 15
    if projection == "UMAP":
        n_neighbors = st.slider("UMAP n_neighbors", 2, 50, 15)

    st.divider()

    # Word selection
    st.subheader("Word selection")
    preset_name = st.selectbox("Preset group", ["(custom)"] + list(PRESETS.keys()))

    if preset_name != "(custom)":
        default_words = PRESETS[preset_name]
    else:
        default_words = PRESETS["Jobs"]

    custom_input = st.text_area(
        "Custom words (one per line)",
        value="\n".join(default_words),
        height=200,
        help="Enter one word per line. Only words in the GloVe vocabulary will be plotted.",
    )
    selected_words = [w.strip() for w in custom_input.splitlines() if w.strip()]

    st.divider()

    # Gender anchors
    st.subheader("Gender anchors")
    anchor_man_words = st.text_input(
        "Masculine anchor words (comma-separated)",
        value="man",
    )
    anchor_woman_words = st.text_input(
        "Feminine anchor words (comma-separated)",
        value="woman",
    )

# ── Resolve anchors ──────────────────────────────────────────────────────────
man_words = [w.strip().lower() for w in anchor_man_words.split(",") if w.strip()]
woman_words = [w.strip().lower() for w in anchor_woman_words.split(",") if w.strip()]

man_vecs = [glove[w] for w in man_words if w in glove]
woman_vecs = [glove[w] for w in woman_words if w in glove]

if not man_vecs or not woman_vecs:
    st.error("At least one masculine and one feminine anchor word must be in the GloVe vocabulary.")
    st.stop()

anchor_man = np.mean(man_vecs, axis=0)
anchor_woman = np.mean(woman_vecs, axis=0)

# ── Resolve selected words ───────────────────────────────────────────────────
valid_words, vectors = get_vectors(selected_words, glove)

if len(valid_words) < 2:
    st.warning("Need at least 2 words in the GloVe vocabulary to plot.")
    st.stop()

skipped = set(selected_words) - set(valid_words)
if skipped:
    st.info(f"Skipped {len(skipped)} words not in vocabulary: {', '.join(sorted(skipped))}")

# ── Bias scores for colouring ────────────────────────────────────────────────
bias_scores = compute_bias_scores(valid_words, vectors, anchor_man, anchor_woman)

# ── Projection ───────────────────────────────────────────────────────────────
if projection == "PCA":
    # Include anchors in projection so we can plot them
    all_vecs = np.vstack([vectors, anchor_man.reshape(1, -1), anchor_woman.reshape(1, -1)])
    all_coords = project_pca(all_vecs)
    coords = all_coords[:len(valid_words)]
    man_coord = all_coords[len(valid_words)]
    woman_coord = all_coords[len(valid_words) + 1]
    x_label, y_label = "PC1", "PC2"
elif projection == "t-SNE":
    all_vecs = np.vstack([vectors, anchor_man.reshape(1, -1), anchor_woman.reshape(1, -1)])
    all_coords = project_tsne(all_vecs, perplexity)
    coords = all_coords[:len(valid_words)]
    man_coord = all_coords[len(valid_words)]
    woman_coord = all_coords[len(valid_words) + 1]
    x_label, y_label = "t-SNE 1", "t-SNE 2"
elif projection == "UMAP":
    all_vecs = np.vstack([vectors, anchor_man.reshape(1, -1), anchor_woman.reshape(1, -1)])
    coords_umap = project_umap(all_vecs, n_neighbors)
    if coords_umap is None:
        st.error("UMAP requires `umap-learn`. Install it: `uv add umap-learn`")
        st.stop()
    coords = coords_umap[:len(valid_words)]
    man_coord = coords_umap[len(valid_words)]
    woman_coord = coords_umap[len(valid_words) + 1]
    x_label, y_label = "UMAP 1", "UMAP 2"
elif projection == "Gender Axis":
    coords, v_gender, v_orth = project_gender_axis(vectors, anchor_man, anchor_woman)
    man_coord = None  # handled separately
    woman_coord = None
    x_label = "← woman ··· gender axis ··· man →"
    y_label = "orthogonal"
else:
    st.error(f"Unknown projection: {projection}")
    st.stop()

# ── Build plot dataframe ─────────────────────────────────────────────────────
df_plot = pd.DataFrame({
    "word": valid_words,
    "x": coords[:, 0],
    "y": coords[:, 1],
    "bias": bias_scores,
    "sim_man": [cosine_similarity(v, anchor_man) for v in vectors],
    "sim_woman": [cosine_similarity(v, anchor_woman) for v in vectors],
})

# Normalise bias for colour scale: centred at 0
bias_max = max(abs(df_plot["bias"].min()), abs(df_plot["bias"].max()), 0.01)

# ── Main layout ──────────────────────────────────────────────────────────────
col_plot, col_analysis = st.columns([3, 2])

with col_plot:
    st.subheader(f"2D Projection — {projection}")

    # Scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot["x"],
        y=df_plot["y"],
        mode="markers+text",
        text=df_plot["word"],
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(
            size=12,
            color=df_plot["bias"],
            colorscale="RdBu",
            cmid=0,
            cmin=-bias_max,
            cmax=bias_max,
            colorbar=dict(title="Bias<br>(+man, −woman)"),
            line=dict(width=1, color="black"),
        ),
        customdata=np.stack([df_plot["sim_man"], df_plot["sim_woman"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sim to man: %{customdata[0]:.3f}<br>"
            "Sim to woman: %{customdata[1]:.3f}<br>"
            "Bias: %{marker.color:.3f}"
            "<extra></extra>"
        ),
    ))

    # Gender axis annotation
    if projection == "Gender Axis":
        x_range = df_plot["x"].max() - df_plot["x"].min()
        fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig.add_annotation(x=df_plot["x"].min() - 0.05 * x_range, y=0,
                           text="← woman", showarrow=False, font=dict(size=14, color="red"))
        fig.add_annotation(x=df_plot["x"].max() + 0.05 * x_range, y=0,
                           text="man →", showarrow=False, font=dict(size=14, color="blue"))

    # Anchor markers for PCA / t-SNE / UMAP
    if projection in ("PCA", "t-SNE", "UMAP"):
        fig.add_trace(go.Scatter(
            x=[man_coord[0]], y=[man_coord[1]],
            mode="markers+text", text=["♂ MAN"], textposition="top center",
            marker=dict(size=20, color="#4a90d9", symbol="star", line=dict(width=2, color="black")),
            textfont=dict(size=16, color="#4a90d9", family="Arial Black"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[woman_coord[0]], y=[woman_coord[1]],
            mode="markers+text", text=["♀ WOMAN"], textposition="top center",
            marker=dict(size=20, color="#d94a6b", symbol="star", line=dict(width=2, color="black")),
            textfont=dict(size=16, color="#d94a6b", family="Arial Black"),
            showlegend=False,
        ))
        # Arrow from woman to man
        fig.add_annotation(
            x=man_coord[0], y=man_coord[1],
            ax=woman_coord[0], ay=woman_coord[1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=4,
            arrowcolor="rgba(100,100,100,0.5)",
            opacity=0.7,
        )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_analysis:
    st.subheader("Bias Analysis")

    # Bar chart: sim to man vs woman
    df_bar = df_plot[["word", "sim_man", "sim_woman"]].melt(
        id_vars="word", var_name="anchor", value_name="cosine_similarity"
    )
    df_bar["anchor"] = df_bar["anchor"].map({"sim_man": "Man/Male", "sim_woman": "Woman/Female"})

    fig_bar = px.bar(
        df_bar,
        x="word",
        y="cosine_similarity",
        color="anchor",
        barmode="group",
        color_discrete_map={"Man/Male": "#4a90d9", "Woman/Female": "#d94a6b"},
        labels={"cosine_similarity": "Cosine Similarity", "word": ""},
    )
    fig_bar.update_layout(
        height=400,
        template="plotly_white",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Table
    df_table = bias_table(valid_words, vectors, anchor_man, anchor_woman)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "word": st.column_config.TextColumn("Word"),
            "sim_man": st.column_config.NumberColumn("Sim → Man", format="%.4f"),
            "sim_woman": st.column_config.NumberColumn("Sim → Woman", format="%.4f"),
            "difference": st.column_config.NumberColumn("Bias (M−W)", format="%.4f"),
        },
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "GloVe embeddings: Pennington, Socher & Manning (2014). "
    f"Vectors are {dim}-dimensional, trained on 6B tokens (Wikipedia 2014 + Gigaword 5)."
)
