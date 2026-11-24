"""All static figures for the report (saved as PNG in reports/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import (dendrogram, linkage,  # noqa: E402
                                     set_link_color_palette)

from src.config import FEATURES, FIGURES  # noqa: E402

# Validated categorical palette (fixed order, never cycled). Colour follows
# the cluster name, so every chart paints the same cluster the same way.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLUSTER_ORDER = ["Upbeat & Energetic", "Mellow Acoustic",
                 "Spoken Word & Audio Drama", "Instrumental & Soundtrack"]
CLUSTER_COLORS = dict(zip(CLUSTER_ORDER, SERIES))
BLUE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8302f"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "legend.frameon": False, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def color_for(name: str) -> str:
    return CLUSTER_COLORS.get(name, "#8a8986")


def _save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


# ----------------------------------------------------------------- EDA ----
def feature_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
    for ax, f in zip(axes.flat, FEATURES):
        vals = df[f] / 60_000 if f == "duration_ms" else df[f]
        if f == "duration_ms":
            vals = vals.clip(upper=12)
        ax.hist(vals, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
        ax.set_title("duration (min, clipped at 12)" if f == "duration_ms" else f,
                     fontsize=10.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distribution of the 10 audio features (94,260 songs)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "01_feature_distributions.png")


def correlation_heatmap(df: pd.DataFrame):
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.grid(False)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Feature correlations: energy, loudness and acousticness move together")
    _save(fig, "02_correlation_heatmap.png")


# ------------------------------------------------------- choosing k -----
def elbow_plot(k_df: pd.DataFrame, k_final: int):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(k_df["k"], k_df["inertia"] / 1e3, color=BLUE, lw=2, marker="o", ms=8)
    sel = k_df[k_df.k == k_final].iloc[0]
    ax.scatter([k_final], [sel.inertia / 1e3], s=170, facecolor="none",
               edgecolor=INK, lw=1.6, zorder=5)
    ax.annotate(f"chosen k = {k_final}", (k_final, sel.inertia / 1e3),
                xytext=(14, 14), textcoords="offset points", color=INK)
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("inertia / SSE (thousands)")
    ax.set_title("Elbow method: gains flatten after k = 3 to 4")
    _save(fig, "03_elbow_method.png")


def k_scores_plot(k_df: pd.DataFrame, k_final: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, title in [
        (axes[0], "silhouette", "Silhouette score (higher is better)"),
        (axes[1], "davies_bouldin", "Davies-Bouldin index (lower is better)"),
    ]:
        ax.plot(k_df["k"], k_df[col], color=BLUE, lw=2, marker="o", ms=8)
        v = k_df.loc[k_df.k == k_final, col].iloc[0]
        ax.scatter([k_final], [v], s=170, facecolor="none", edgecolor=INK, lw=1.6, zorder=5)
        ax.annotate(f"k={k_final}: {v:.3f}", (k_final, v), xytext=(12, 10),
                    textcoords="offset points", color=INK)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("number of clusters (k)")
    fig.tight_layout()
    _save(fig, "04_silhouette_davies_bouldin.png")


# --------------------------------------------------- cluster visuals ----
def _scatter_by_cluster(ax, xy, names, size=4, alpha=0.45):
    for name in CLUSTER_ORDER:
        m = names == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, alpha=alpha,
                       color=color_for(name), label=name, linewidths=0)
    leg = ax.legend(markerscale=4, loc="best", fontsize=9)
    for h in leg.legend_handles:
        h.set_alpha(1)


def pca_scatter(pcs: np.ndarray, names: np.ndarray, explained: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_by_cluster(ax, pcs, names, size=3, alpha=0.35)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title("K-Means clusters projected onto the first two principal components")
    _save(fig, "05_pca_clusters.png")


def tsne_scatter(emb: np.ndarray, names: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_by_cluster(ax, emb, names, size=6, alpha=0.6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE view of a 6,000-song sample, coloured by K-Means cluster")
    _save(fig, "06_tsne_clusters.png")


def profile_heatmap(prof: pd.DataFrame, overall: pd.DataFrame):
    """Cluster means expressed as z-scores against the whole dataset."""
    z = (prof[FEATURES] - overall[FEATURES].mean()) / overall[FEATURES].std()
    z = z.loc[[n for n in CLUSTER_ORDER if n in z.index]]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    im = ax.imshow(z.values, cmap=DIVERGING, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=30, ha="right")
    ax.set_yticks(range(len(z)), z.index)
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            raw = prof.loc[z.index[i], FEATURES[j]]
            txt = f"{raw/60000:.1f}m" if FEATURES[j] == "duration_ms" else (
                f"{raw:.0f}" if abs(raw) >= 10 else f"{raw:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if abs(z.values[i, j]) > 1.6 else INK)
    fig.colorbar(im, ax=ax, label="z-score vs all songs", shrink=0.9)
    ax.set_title("Cluster profiles: cell text = cluster mean, colour = how far from the overall mean")
    _save(fig, "07_cluster_profile_heatmap.png")


def feature_means_bars(prof: pd.DataFrame):
    feats = ["danceability", "energy", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence"]
    names = [n for n in CLUSTER_ORDER if n in prof.index]
    x = np.arange(len(feats))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, prof.loc[n, feats], w * 0.92,
               color=color_for(n), label=n)
    ax.set_xticks(x, feats)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean value (0 to 1 scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax.set_title("Average 0-1 audio features per cluster", pad=26)
    _save(fig, "08_feature_means_by_cluster.png")


def feature_boxplots(df: pd.DataFrame):
    feats = ["danceability", "energy", "valence", "acousticness", "tempo", "duration_ms"]
    names = [n for n in CLUSTER_ORDER if n in set(df.cluster_name)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.flat, feats):
        data = [(df.loc[df.cluster_name == n, f] / (60_000 if f == "duration_ms" else 1))
                .clip(upper=10 if f == "duration_ms" else None) for n in names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": INK, "lw": 1.5})
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(color_for(n))
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, len(names) + 1),
                      [n.replace(" & ", "\n& ") for n in names], fontsize=8.5)
        ax.set_title("duration (min)" if f == "duration_ms" else f, fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Feature distributions within each cluster (outliers hidden)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "09_feature_boxplots_by_cluster.png")


def cluster_sizes(prof: pd.DataFrame):
    p = prof.loc[[n for n in CLUSTER_ORDER if n in prof.index]].sort_values("songs")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(p.index, p["songs"], color=[color_for(n) for n in p.index], height=0.6)
    for i, (n, row) in enumerate(p.iterrows()):
        ax.text(row.songs + 600, i, f"{row.songs:,.0f} ({row.share_pct:.1f}%)",
                va="center", color=INK, fontsize=9.5)
    ax.set_xlim(0, p["songs"].max() * 1.25)
    ax.set_xlabel("songs")
    ax.grid(axis="y", visible=False)
    ax.set_title("Cluster sizes")
    _save(fig, "10_cluster_sizes.png")


def dendrogram_plot(Xsample: np.ndarray, k: int):
    Z = linkage(Xsample, method="ward")
    set_link_color_palette([BLUE])
    fig, ax = plt.subplots(figsize=(12, 5))
    cut = Z[-(k - 1), 2] - 1e-6
    dendrogram(Z, truncate_mode="lastp", p=40, ax=ax, color_threshold=cut,
               above_threshold_color="#8a8986", no_labels=True)
    ax.axhline(cut, color=INK, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], cut, f"  cut for {k} clusters", va="bottom",
            ha="right", color=INK)
    ax.set_ylabel("Ward linkage distance")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Hierarchical (Ward) dendrogram on a {len(Xsample):,}-song sample")
    _save(fig, "11_dendrogram.png")


def k_distance_plot(kd: np.ndarray, eps: float, k: int):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(np.arange(len(kd)), kd, color=BLUE, lw=2)
    ax.axhline(eps, color=INK, ls="--", lw=1)
    ax.text(len(kd) * 0.02, eps, f" eps = {eps}", va="bottom", color=INK)
    ax.set_xlabel("songs sorted by distance")
    ax.set_ylabel(f"distance to {k}th nearest neighbour")
    ax.set_title("DBSCAN k-distance plot: no clear knee, density is continuous")
    _save(fig, "12_dbscan_k_distance.png")


def algorithm_comparison(cmp_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    order = cmp_df["algorithm"].tolist()[::-1]
    d = cmp_df.set_index("algorithm").loc[order]
    for ax, col, title in [(axes[0], "silhouette",
                            "Silhouette (higher is better; DBSCAN scored without its noise points)"),
                           (axes[1], "largest_cluster_pct",
                            "Largest cluster, % of songs (lower = more balanced)")]:
        colors = [BLUE if a == "K-Means" else "#9ec5f4" for a in d.index]
        ax.barh(d.index, d[col], color=colors, height=0.55)
        for i, v in enumerate(d[col]):
            ax.text(v, i, f"  {v:.3f}" if col == "silhouette" else f"  {v:.1f}%",
                    va="center", color=INK, fontsize=9.5)
        ax.set_xlim(0, d[col].max() * 1.3)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "13_algorithm_comparison.png")


def genre_heatmap(gdf: pd.DataFrame):
    """Share of each top genre that falls in each cluster."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    names = [n for n in CLUSTER_ORDER if n in gdf.columns]
    m = gdf[names]
    im = ax.imshow(m.values, cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(names)), [n.replace(" & ", "\n& ") for n in names], fontsize=9)
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 55 else INK)
    fig.colorbar(im, ax=ax, label="% of the genre's songs", shrink=0.8)
    ax.set_title("Where the 15 biggest genres land: spoken word and vintage pop separate cleanly,\nmost pop genres split between upbeat and mellow")
    _save(fig, "14_genre_by_cluster.png")
