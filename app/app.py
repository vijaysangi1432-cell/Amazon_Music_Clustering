"""Amazon Music Clustering: Streamlit app.

Run from the project root:  streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURES, FIGURES, MODELS, OUTPUT_CSV, REPORTS  # noqa: E402
from src.clustering import CLUSTER_DESCRIPTIONS  # noqa: E402
from src.visualize import CLUSTER_ORDER, SERIES  # noqa: E402

st.set_page_config(page_title="Amazon Music Clustering", page_icon="🎵", layout="wide")

COLOR_SCALE = alt.Scale(domain=CLUSTER_ORDER, range=SERIES)
PLAYLIST_IDEAS = {
    "Upbeat & Energetic": "Party · Workout · Driving",
    "Mellow Acoustic": "Chill · Relax · Nostalgia",
    "Spoken Word & Audio Drama": "Audiobooks · Radio plays · Comedy",
    "Instrumental & Soundtrack": "Focus · Study · Sleep",
}


# ------------------------------------------------------------- data -------
@st.cache_data
def load_songs() -> pd.DataFrame:
    return pd.read_csv(OUTPUT_CSV)


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODELS / "kmeans_model.joblib")


@st.cache_data
def load_reports():
    return (pd.read_csv(REPORTS / "cluster_profiles.csv", index_col=0),
            pd.read_csv(REPORTS / "k_selection.csv"),
            pd.read_csv(REPORTS / "algorithm_comparison.csv"),
            json.loads((REPORTS / "summary.json").read_text()))


def to_matrix(frame: pd.DataFrame, model: dict) -> np.ndarray:
    X = frame[FEATURES].copy()
    X["duration_ms"] = np.log1p(X["duration_ms"])
    return model["scaler"].transform(X)


@st.cache_data
def scaled_matrix() -> np.ndarray:
    return to_matrix(load_songs(), load_model())


def nearest_songs(vec: np.ndarray, cluster: int, n: int = 10, exclude=None,
                  exclude_pair=None) -> pd.DataFrame:
    df = load_songs()
    Xs = scaled_matrix()
    mask = (df["cluster"] == cluster).to_numpy().copy()
    if exclude is not None:
        mask[exclude] = False
    d = np.linalg.norm(Xs[mask] - vec, axis=1)
    out = df.loc[mask, ["name_song", "name_artists", "genre", "popularity_songs",
                        "energy", "danceability", "acousticness", "valence"]].copy()
    out["distance"] = d
    if exclude_pair is not None:  # drop other releases of the chosen song
        out = out[~((out.name_song == exclude_pair[0])
                    & (out.name_artists == exclude_pair[1]))]
    out = out.sort_values("distance").drop_duplicates(["name_song", "name_artists"])
    return out.head(n).reset_index(drop=True)


songs = load_songs()
model = load_model()
profiles, k_sel, algo_cmp, summary = load_reports()

# ------------------------------------------------------------ layout ------
st.sidebar.title("🎵 Amazon Music Clustering")
page = st.sidebar.radio("Go to", ["Overview", "Cluster explorer", "Find similar songs",
                                  "Predict a cluster", "Model evaluation"])
st.sidebar.caption(f"{len(songs):,} songs · K-Means k = {summary['k_final']}")

# ------------------------------------------------------------ overview ----
if page == "Overview":
    st.title("Songs grouped by how they sound")
    st.write("K-Means on 10 audio features groups the catalogue into four clusters, "
             "without using any genre labels.")
    km = summary["kmeans_full"]
    c = st.columns(4)
    c[0].metric("Songs clustered", f"{len(songs):,}")
    c[1].metric("Clusters", summary["k_final"])
    c[2].metric("Silhouette score", f"{km['silhouette_sample_20k']:.3f}")
    c[3].metric("Davies-Bouldin index", f"{km['davies_bouldin']:.3f}")

    st.subheader("The four clusters")
    cols = st.columns(4)
    for col, name in zip(cols, CLUSTER_ORDER):
        r = profiles.loc[name]
        with col:
            st.markdown(f"<div style='border-left:5px solid "
                        f"{SERIES[CLUSTER_ORDER.index(name)]};padding-left:10px'>"
                        f"<b>{name}</b><br><span style='font-size:1.6em'>"
                        f"{int(r.songs):,}</span> songs ({r.share_pct:.1f}%)</div>",
                        unsafe_allow_html=True)
            st.caption(CLUSTER_DESCRIPTIONS[name])
            st.caption(f"**Playlist ideas:** {PLAYLIST_IDEAS[name]}")

    st.subheader("Clusters in 2D (PCA, 8,000-song sample)")
    sample = songs.sample(8000, random_state=1)
    chart = (alt.Chart(sample).mark_circle(size=14, opacity=0.55).encode(
        x=alt.X("pc1:Q", title="PC1"), y=alt.Y("pc2:Q", title="PC2"),
        color=alt.Color("cluster_name:N", scale=COLOR_SCALE, title="Cluster",
                        legend=alt.Legend(orient="top")),
        tooltip=["name_song", "name_artists", "genre", "cluster_name",
                 alt.Tooltip("energy:Q", format=".2f"),
                 alt.Tooltip("acousticness:Q", format=".2f")])
        .properties(height=480).interactive())
    st.altair_chart(chart, width="stretch")

# --------------------------------------------------- cluster explorer -----
elif page == "Cluster explorer":
    name = st.selectbox("Cluster", CLUSTER_ORDER)
    sub = songs[songs.cluster_name == name]
    st.title(name)
    st.write(CLUSTER_DESCRIPTIONS[name])
    c = st.columns(4)
    c[0].metric("Songs", f"{len(sub):,}")
    c[1].metric("Share of catalogue", f"{100 * len(sub) / len(songs):.1f}%")
    c[2].metric("Avg duration", f"{sub.duration_ms.mean() / 60000:.1f} min")
    c[3].metric("Avg tempo", f"{sub.tempo.mean():.0f} BPM")

    st.subheader("Audio profile vs the whole catalogue")
    feats = ["danceability", "energy", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence"]
    comp = pd.DataFrame({"feature": feats * 2,
                         "group": [name] * len(feats) + ["All songs"] * len(feats),
                         "mean": list(sub[feats].mean()) + list(songs[feats].mean())})
    bar = (alt.Chart(comp).mark_bar(cornerRadiusEnd=4, height=12).encode(
        y=alt.Y("feature:N", sort=feats, title=None),
        yOffset=alt.YOffset("group:N", sort=[name, "All songs"]),
        x=alt.X("mean:Q", title="mean (0 to 1)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", scale=alt.Scale(
            domain=[name, "All songs"],
            range=[SERIES[CLUSTER_ORDER.index(name)], "#c3c2b7"]),
            title=None, legend=alt.Legend(orient="top")),
        tooltip=["group", "feature", alt.Tooltip("mean:Q", format=".3f")])
        .properties(height=360))
    st.altair_chart(bar, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Most popular songs")
        query = st.text_input("Filter by song or artist", "")
        view = sub
        if query:
            q = query.lower()
            view = sub[sub.name_song.str.lower().str.contains(q, regex=False)
                       | sub.name_artists.str.lower().str.contains(q, regex=False)]
        st.dataframe(view.sort_values("popularity_songs", ascending=False)
                     .drop_duplicates(["name_song", "name_artists"])
                     [["name_song", "name_artists", "genre", "popularity_songs",
                       "energy", "acousticness"]].head(50),
                     hide_index=True, width="stretch", height=420)
    with right:
        st.subheader("Top genres in this cluster")
        g = sub.genre.value_counts().head(12).rename_axis("genre").reset_index(name="songs")
        gchart = (alt.Chart(g).mark_bar(cornerRadiusEnd=4,
                                        color=SERIES[CLUSTER_ORDER.index(name)]).encode(
            y=alt.Y("genre:N", sort="-x", title=None), x=alt.X("songs:Q"),
            tooltip=["genre", "songs"]).properties(height=420))
        st.altair_chart(gchart, width="stretch")
    st.download_button(f"Download {name} songs (CSV)",
                       sub.to_csv(index=False).encode("utf-8"),
                       file_name=f"cluster_{name.split()[0].lower()}.csv")

# -------------------------------------------------- similar songs ---------
elif page == "Find similar songs":
    st.title("Find similar songs")
    st.write("Pick a song. The app finds its cluster and the closest-sounding "
             "songs in the same cluster (Euclidean distance on scaled audio features).")
    q = st.text_input("Search a song or artist", "Arcade")
    hits = songs[songs.name_song.str.contains(q, case=False, regex=False)
                 | songs.name_artists.str.contains(q, case=False, regex=False)] if q else songs.head(0)
    hits = hits.sort_values("popularity_songs", ascending=False).head(200)
    if hits.empty:
        st.info("No match. Try another title or artist.")
    else:
        label = hits.apply(lambda r: f"{r.name_song} · {r.name_artists} ({r.genre})", axis=1)
        choice = st.selectbox("Matches", label.index, format_func=lambda i: label[i])
        song = songs.loc[choice]
        st.markdown(f"**{song.name_song}** by {song.name_artists} → cluster "
                    f"**{song.cluster_name}**")
        vec = scaled_matrix()[choice]
        n = st.slider("How many recommendations", 5, 25, 10)
        st.dataframe(nearest_songs(vec, int(song.cluster), n, exclude=choice,
                                   exclude_pair=(song.name_song, song.name_artists)),
                     hide_index=True, width="stretch")

# ------------------------------------------------------- predict ----------
elif page == "Predict a cluster":
    st.title("Which cluster would a new song fall into?")
    st.write("Set the audio features of a track. The trained scaler + K-Means "
             "model assigns it to a cluster.")
    med = songs[FEATURES].median()
    c1, c2 = st.columns(2)
    vals = {}
    with c1:
        for f in ["danceability", "energy", "speechiness", "acousticness", "instrumentalness"]:
            vals[f] = st.slider(f, 0.0, 1.0, float(round(med[f], 2)), 0.01)
    with c2:
        for f in ["liveness", "valence"]:
            vals[f] = st.slider(f, 0.0, 1.0, float(round(med[f], 2)), 0.01)
        vals["loudness"] = st.slider("loudness (dB)", -40.0, 0.0, float(round(med.loudness, 1)), 0.5)
        vals["tempo"] = st.slider("tempo (BPM)", 40.0, 220.0, float(round(med.tempo)), 1.0)
        vals["duration_ms"] = st.slider("duration (min)", 0.5, 12.0, 3.5, 0.1) * 60_000
    row = pd.DataFrame([vals])[FEATURES]
    vec = to_matrix(row, model)[0]
    cl_id = int(model["kmeans"].predict(vec.reshape(1, -1))[0])
    name = model["names"][cl_id]
    dists = np.linalg.norm(model["kmeans"].cluster_centers_ - vec, axis=1)
    st.success(f"Predicted cluster: **{name}**. Playlist ideas: {PLAYLIST_IDEAS.get(name, '')}")
    dd = pd.DataFrame({"cluster": [model["names"][i] for i in range(len(dists))],
                       "distance": dists})
    st.altair_chart(alt.Chart(dd).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=alt.Y("cluster:N", sort="x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("distance:Q", title="distance to cluster centre (lower = closer)"),
        color=alt.Color("cluster:N", scale=COLOR_SCALE, legend=None),
        tooltip=["cluster", alt.Tooltip("distance:Q", format=".2f")]).properties(height=200),
        width="stretch")
    st.subheader("Closest existing songs")
    st.dataframe(nearest_songs(vec, cl_id, 10), hide_index=True, width="stretch")

# ---------------------------------------------------- evaluation ----------
else:
    st.title("Model evaluation")
    st.write(f"k = 3 has the best silhouette, but k = {summary['k_final']} is close behind, "
             "has the lowest Davies-Bouldin index and separates instrumental music, "
             "so k = 4 was chosen.")
    k_long = k_sel.melt("k", ["inertia", "silhouette", "davies_bouldin"])
    titles = {"inertia": "Inertia / SSE (elbow)", "silhouette": "Silhouette (higher = better)",
              "davies_bouldin": "Davies-Bouldin (lower = better)"}
    cols = st.columns(3)
    for col, m in zip(cols, titles):
        d = k_long[k_long.variable == m]
        base = alt.Chart(d).encode(x=alt.X("k:O", title="k"),
                                   y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)))
        line = base.mark_line(strokeWidth=2, color=SERIES[0]) + base.mark_point(
            filled=True, size=70, color=SERIES[0]).encode(
            tooltip=["k", alt.Tooltip("value:Q", format=".4f")])
        pick = base.transform_filter(alt.datum.k == summary["k_final"]).mark_point(
            size=260, color="#0b0b0b", strokeWidth=2)
        col.markdown(f"**{titles[m]}**")
        col.altair_chart((line + pick).properties(height=260), width="stretch")

    st.subheader("Algorithm comparison (same 10,000-song sample)")
    st.dataframe(algo_cmp.round(3), hide_index=True, width="stretch")
    st.caption("DBSCAN scores only its non-noise points: it calls about a third of songs "
               "noise and puts about 91% of the rest in one cluster, so it does not "
               "segment the catalogue.")
    st.subheader("Report figures")
    figs = sorted(FIGURES.glob("*.png"))
    pick = st.selectbox("Figure", figs, format_func=lambda p: p.stem.replace("_", " "))
    st.image(str(pick), width="stretch")
