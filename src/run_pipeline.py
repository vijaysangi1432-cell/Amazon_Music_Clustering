"""End-to-end pipeline: clean -> scale -> choose k -> cluster -> evaluate ->
interpret -> visualise -> export.

Run from the project root:  python -m src.run_pipeline
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src import clustering as cl
from src import visualize as viz
from src.config import (DBSCAN_EPS, DBSCAN_EPS_GRID, DBSCAN_MIN_SAMPLES,
                        DENDRO_SAMPLE, FEATURES, K_FINAL, MODELS, OUTPUT_CSV,
                        PROCESSED_CSV, RANDOM_STATE, REFERENCE_COLS, REPORTS,
                        ROOT, TSNE_SAMPLE)
from src.data_prep import clean, feature_matrix, inspect, load_data, scale


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    for p in (REPORTS, MODELS, OUTPUT_CSV.parent, PROCESSED_CSV.parent):
        p.mkdir(parents=True, exist_ok=True)

    # 1. Explore & clean --------------------------------------------------
    raw = load_data()
    facts = inspect(raw)
    log(f"Loaded {facts['rows']:,} rows x {facts['columns']} columns")
    df, clean_log = clean(raw)
    log(f"Cleaned: {clean_log}")
    df.to_csv(PROCESSED_CSV, index=False)
    viz.feature_distributions(df)
    viz.correlation_heatmap(df)

    # 2-3. Feature selection & scaling ------------------------------------
    X = feature_matrix(df)
    Xs, scaler = scale(X)

    # 4. Choose k ---------------------------------------------------------
    log("Evaluating k = 2..10 (elbow, silhouette, Davies-Bouldin)")
    k_df = cl.evaluate_k_range(Xs)
    k_df.to_csv(REPORTS / "k_selection.csv", index=False)
    print(k_df.round(4).to_string(index=False))
    viz.elbow_plot(k_df, K_FINAL)
    viz.k_scores_plot(k_df, K_FINAL)

    # 5. Final K-Means ----------------------------------------------------
    km = cl.fit_kmeans(Xs, K_FINAL)
    df["cluster"] = km.labels_
    prof = cl.profile(df)
    names = {c: cl.name_cluster(prof.loc[c]) for c in prof.index}
    if len(set(names.values())) < len(names):  # keep names unique
        names = {c: f"{n} ({c})" for c, n in names.items()}
    df["cluster_name"] = df["cluster"].map(names)
    prof.index = prof.index.map(names)
    prof.index.name = "cluster_name"
    log(f"Cluster names: {names}")

    # 6. Alternative algorithms -------------------------------------------
    log("DBSCAN k-distance + eps grid, Agglomerative, comparison")
    idx = np.random.RandomState(7).choice(len(Xs), 10_000, replace=False)
    kd = cl.k_distance(Xs[idx], DBSCAN_MIN_SAMPLES)
    viz.k_distance_plot(kd, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    grid = [{"eps": e, "min_samples": DBSCAN_MIN_SAMPLES,
             **cl.scores(Xs[idx], DBSCAN(eps=e, min_samples=DBSCAN_MIN_SAMPLES)
                         .fit_predict(Xs[idx]))} for e in DBSCAN_EPS_GRID]
    pd.DataFrame(grid).to_csv(REPORTS / "dbscan_eps_grid.csv", index=False)
    cmp_df, _ = cl.compare_algorithms(Xs, km, K_FINAL, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    cmp_df.to_csv(REPORTS / "algorithm_comparison.csv", index=False)
    print(cmp_df.round(4).to_string(index=False))
    viz.algorithm_comparison(cmp_df)
    didx = np.random.RandomState(3).choice(len(Xs), DENDRO_SAMPLE, replace=False)
    viz.dendrogram_plot(Xs[didx], K_FINAL)

    # 7. Visualise clusters -----------------------------------------------
    pca = PCA(n_components=3, random_state=RANDOM_STATE).fit(Xs)
    pcs = pca.transform(Xs)
    df["pc1"], df["pc2"], df["pc3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]
    viz.pca_scatter(pcs, df["cluster_name"].values, pca.explained_variance_ratio_)
    pd.DataFrame(pca.components_, columns=FEATURES,
                 index=["PC1", "PC2", "PC3"]).to_csv(REPORTS / "pca_loadings.csv")
    log("t-SNE on a sample")
    tidx = np.random.RandomState(11).choice(len(Xs), TSNE_SAMPLE, replace=False)
    emb = TSNE(n_components=2, perplexity=40, init="pca",
               random_state=RANDOM_STATE).fit_transform(Xs[tidx])
    viz.tsne_scatter(emb, df["cluster_name"].values[tidx])
    viz.profile_heatmap(prof, df)
    viz.feature_means_bars(prof)
    viz.feature_boxplots(df)
    viz.cluster_sizes(prof)

    # 8. Interpretation tables --------------------------------------------
    prof["description"] = [cl.CLUSTER_DESCRIPTIONS.get(n.split(" (")[0], "")
                           for n in prof.index]
    prof.round(4).to_csv(REPORTS / "cluster_profiles.csv")
    cl.top_genres(df, 10).to_csv(REPORTS / "top_genres_per_cluster.csv", index=False)
    cl.top_tracks(df, 15).to_csv(REPORTS / "top_tracks_per_cluster.csv", index=False)
    cl.typical_tracks(df, Xs, km, 10).to_csv(REPORTS / "typical_tracks_per_cluster.csv",
                                             index=False)
    top15 = df["genre"].value_counts().drop("unknown", errors="ignore").head(15).index
    gx = (pd.crosstab(df.loc[df.genre.isin(top15), "genre"],
                      df.loc[df.genre.isin(top15), "cluster_name"],
                      normalize="index") * 100).loc[top15]
    gx.to_csv(REPORTS / "genre_by_cluster_pct.csv")
    viz.genre_heatmap(gx)
    # how "pure" each genre is: share of its songs in its main cluster
    purity = (pd.crosstab(df.genre, df.cluster_name).pipe(
        lambda t: (t.max(axis=1) / t.sum(axis=1))[t.sum(axis=1) >= 50]))

    # 9. Export -----------------------------------------------------------
    out_cols = REFERENCE_COLS + ["genre"] + FEATURES + [
        "key", "mode", "time_signature", "cluster", "cluster_name", "pc1", "pc2"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)
    joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "names": names,
                 "features": FEATURES}, MODELS / "kmeans_model.joblib", compress=3)

    sel = k_df.set_index("k").loc[K_FINAL]
    best_sil_k = int(k_df.loc[k_df.silhouette.idxmax(), "k"])
    summary = {
        "data": facts, "cleaning": clean_log, "k_final": K_FINAL,
        "best_silhouette_k": best_sil_k,
        "kmeans_full": {"silhouette_sample_20k": round(sel.silhouette, 4),
                        "davies_bouldin": round(sel.davies_bouldin, 4),
                        "calinski_harabasz": round(sel.calinski_harabasz, 1),
                        "inertia": round(sel.inertia, 1)},
        "pca_explained_variance": [round(v, 4) for v in pca.explained_variance_ratio_],
        "clusters": {n: {"songs": int(r.songs), "share_pct": round(r.share_pct, 2)}
                     for n, r in prof.iterrows()},
        "genres_with_50plus_songs": int(len(purity)),
        "median_genre_purity_pct": round(100 * float(purity.median()), 1),
        "genres_90pct_in_one_cluster_pct": round(100 * float((purity >= 0.9).mean()), 1),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    update_readme(summary, prof, cmp_df)
    log(f"Done in {summary['runtime_sec']} s. Output: {OUTPUT_CSV.relative_to(ROOT)}")


def update_readme(summary: dict, prof: pd.DataFrame, cmp_df: pd.DataFrame):
    """Rewrite the results block in README.md between the markers."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    km = summary["kmeans_full"]
    lines = [
        f"- Songs after cleaning: **{summary['cleaning']['rows_out']:,}** "
        f"(from {summary['cleaning']['rows_in']:,})",
        f"- Final model: **K-Means, k = {summary['k_final']}**: silhouette "
        f"{km['silhouette_sample_20k']:.3f}, Davies-Bouldin {km['davies_bouldin']:.3f}",
        "", "| Cluster | Songs | Share | Key traits |", "|---|---:|---:|---|",
    ]
    for n, r in prof.sort_values("songs", ascending=False).iterrows():
        lines.append(f"| {n} | {r.songs:,.0f} | {r.share_pct:.1f}% | "
                     f"{r.description.split(':')[0]} |")
    lines += ["", "| Algorithm (10k sample) | Clusters | Silhouette | Davies-Bouldin | Noise |",
              "|---|---:|---:|---:|---:|"]
    for _, r in cmp_df.iterrows():
        lines.append(f"| {r.algorithm} | {r.n_clusters} | {r.silhouette:.3f} | "
                     f"{r.davies_bouldin:.3f} | {r.noise_pct:.1f}% |")
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        post = rest.split(end, 1)[1]
        readme.write_text(pre + start + "\n" + "\n".join(lines) + "\n" + end + post,
                          encoding="utf-8")


if __name__ == "__main__":
    main()
