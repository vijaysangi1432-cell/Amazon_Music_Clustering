"""Clustering, evaluation and cluster interpretation."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors

from src.config import (COMPARE_SAMPLE, FEATURES, K_RANGE, RANDOM_STATE,
                        SILHOUETTE_SAMPLE)


def _sample_idx(n: int, size: int, seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=min(size, n), replace=False)


# --------------------------------------------------------------------------
# K-Means: choosing k
# --------------------------------------------------------------------------
def evaluate_k_range(Xs: np.ndarray, k_range=K_RANGE) -> pd.DataFrame:
    """Fit K-Means for each k and record inertia (elbow), silhouette,
    Davies-Bouldin and Calinski-Harabasz scores plus cluster size balance.

    Silhouette is computed on a fixed random sample because it is O(n^2).
    """
    idx = _sample_idx(len(Xs), SILHOUETTE_SAMPLE)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        sizes = np.bincount(km.labels_)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Xs[idx], km.labels_[idx]),
            "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
            "smallest_cluster_pct": 100 * sizes.min() / sizes.sum(),
            "largest_cluster_pct": 100 * sizes.max() / sizes.sum(),
        })
    return pd.DataFrame(rows)


def fit_kmeans(Xs: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Xs)


# --------------------------------------------------------------------------
# Alternative algorithms (run on a sample: both scale as O(n^2) in memory)
# --------------------------------------------------------------------------
def k_distance(Xs: np.ndarray, k: int = 20) -> np.ndarray:
    """Sorted distance to the k-th neighbour, used to pick DBSCAN eps."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    return np.sort(dist[:, -1])


def scores(Xs: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette / DB / CH on non-noise points, plus size balance."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    out = {"n_clusters": n_clusters,
           "noise_pct": 100 * (~mask).mean()}
    if n_clusters >= 2:
        sizes = np.bincount(labels[mask])
        sizes = sizes[sizes > 0]
        out.update({
            "silhouette": silhouette_score(Xs[mask], labels[mask]),
            "davies_bouldin": davies_bouldin_score(Xs[mask], labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(Xs[mask], labels[mask]),
            "smallest_cluster_pct": 100 * sizes.min() / sizes.sum(),
            "largest_cluster_pct": 100 * sizes.max() / sizes.sum(),
        })
    return out


def compare_algorithms(Xs: np.ndarray, kmeans: KMeans, k: int,
                       eps: float, min_samples: int) -> tuple[pd.DataFrame, dict]:
    """Score K-Means, DBSCAN and Agglomerative (Ward) on the same sample."""
    idx = _sample_idx(len(Xs), COMPARE_SAMPLE, seed=7)
    Xsub = Xs[idx]
    labels = {
        "K-Means": kmeans.predict(Xsub),
        "Agglomerative (Ward)": AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(Xsub),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xsub),
    }
    rows = []
    for name, lab in labels.items():
        rows.append({"algorithm": name, **scores(Xsub, lab)})
    return pd.DataFrame(rows), {"idx": idx, "labels": labels}


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------
def profile(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    """Mean of every audio feature per cluster, plus size."""
    prof = df.groupby(label_col)[FEATURES].mean()
    prof["duration_min"] = prof["duration_ms"] / 60_000
    prof["songs"] = df[label_col].value_counts().sort_index()
    prof["share_pct"] = 100 * prof["songs"] / prof["songs"].sum()
    prof["avg_popularity"] = df.groupby(label_col)["popularity_songs"].mean()
    return prof


def name_cluster(row: pd.Series) -> str:
    """Turn a cluster's mean audio profile into a human-readable label.

    Rules are applied in order, using the features that separate the
    clusters most clearly (see the profile heatmap).
    """
    if row["speechiness"] >= 0.5:
        return "Spoken Word & Audio Drama"
    if row["instrumentalness"] >= 0.5:
        return "Instrumental & Soundtrack"
    if row["energy"] >= 0.6:
        return "Upbeat & Energetic"
    if row["acousticness"] >= 0.6:
        return "Mellow Acoustic"
    return "Mid-tempo Mix"


CLUSTER_DESCRIPTIONS = {
    "Spoken Word & Audio Drama": (
        "Very high speechiness and short duration: audio-book and radio-play "
        "chapters (German Hoerspiel dominates), plus comedy and spoken "
        "tracks. Keep out of music playlists; recommend within itself."),
    "Instrumental & Soundtrack": (
        "Very high instrumentalness, quiet and acoustic: film scores, "
        "classical, ambient and lo-fi pieces. Suited to focus, study and "
        "background playlists."),
    "Upbeat & Energetic": (
        "High energy, loud, danceable and positive (high valence): modern "
        "pop, J-pop, dance and rock. Suited to party, workout and "
        "driving playlists."),
    "Mellow Acoustic": (
        "Low energy, quiet and very acoustic with vocals: vintage and classic "
        "pop, ballads and folk from many countries. Suited to chill, "
        "relaxing and nostalgia playlists."),
    "Mid-tempo Mix": "Average on most features: a general-purpose mix.",
}


def top_genres(df: pd.DataFrame, n: int = 5, label_col: str = "cluster_name"
               ) -> pd.DataFrame:
    """Most common artist genres inside each cluster (post-hoc check)."""
    rows = []
    for name, g in df.groupby(label_col):
        vc = g["genre"].value_counts()
        for genre, cnt in vc.head(n).items():
            rows.append({label_col: name, "genre": genre, "songs": cnt,
                         "share_of_cluster_pct": 100 * cnt / len(g)})
    return pd.DataFrame(rows)


def top_tracks(df: pd.DataFrame, n: int = 10, label_col: str = "cluster_name"
               ) -> pd.DataFrame:
    """Most popular songs per cluster (one song per title+artist)."""
    cols = [label_col, "name_song", "name_artists", "genre", "popularity_songs"]
    out = (df.sort_values("popularity_songs", ascending=False)
             .drop_duplicates(["name_song", "name_artists"])
             .groupby(label_col).head(n)[cols + ["energy", "acousticness",
                                                  "instrumentalness", "speechiness"]]
             .sort_values([label_col, "popularity_songs"],
                          ascending=[True, False]))
    return out.reset_index(drop=True)


def typical_tracks(df: pd.DataFrame, Xs: np.ndarray, kmeans: KMeans,
                   n: int = 10, label_col: str = "cluster_name") -> pd.DataFrame:
    """Songs closest to each centroid: the most 'typical' member of a cluster."""
    dist = np.linalg.norm(Xs - kmeans.cluster_centers_[df["cluster"]], axis=1)
    tmp = df.assign(dist_to_centroid=dist)
    cols = [label_col, "name_song", "name_artists", "genre", "dist_to_centroid"]
    return (tmp.sort_values("dist_to_centroid").groupby(label_col).head(n)[cols]
               .sort_values([label_col, "dist_to_centroid"]).reset_index(drop=True))
