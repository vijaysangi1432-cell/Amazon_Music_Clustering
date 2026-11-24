# Amazon Music Clustering

Unsupervised grouping of ~94,000 Amazon Music songs by their audio characteristics
(danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, duration) using K-Means, with DBSCAN and hierarchical
clustering as comparisons. The clusters are profiled, named and exported for
playlist curation and song discovery.

**Domain:** Music Analytics / Unsupervised Machine Learning
**Skills:** data exploration and cleaning, feature selection, scaling, K-Means, elbow method,
silhouette score, Davies-Bouldin index, PCA, t-SNE, DBSCAN, hierarchical clustering,
cluster visualisation, genre inference, Streamlit

## Results

<!-- RESULTS:START -->
- Songs after cleaning: **94,260** (from 95,837)
- Final model: **K-Means, k = 4**: silhouette 0.232, Davies-Bouldin 1.509

| Cluster | Songs | Share | Key traits |
|---|---:|---:|---|
| Upbeat & Energetic | 45,332 | 48.1% | High energy, loud, danceable and positive (high valence) |
| Mellow Acoustic | 28,777 | 30.5% | Low energy, quiet and very acoustic with vocals |
| Spoken Word & Audio Drama | 12,619 | 13.4% | Very high speechiness and short duration |
| Instrumental & Soundtrack | 7,532 | 8.0% | Very high instrumentalness, quiet and acoustic |

| Algorithm (10k sample) | Clusters | Silhouette | Davies-Bouldin | Noise |
|---|---:|---:|---:|---:|
| K-Means | 4 | 0.234 | 1.504 | 0.0% |
| Agglomerative (Ward) | 4 | 0.197 | 1.661 | 0.0% |
| DBSCAN | 2 | 0.384 | 0.939 | 32.1% |
<!-- RESULTS:END -->

**Why k = 4:** k = 3 has the highest silhouette, but k = 4 is within 0.013 of it,
has the lowest Davies-Bouldin index of all k tried, and splits instrumental and
soundtrack music away from the vocal "mellow" songs, which is a more useful split
for playlists. DBSCAN finds one large dense blob (about 91% of the songs it
clusters) and labels about a third of songs as noise, because song density changes
gradually with no clear gaps. Its higher silhouette only counts the
non-noise points, so it is not a usable segmentation here.

## Project structure

```
Amazon_Music_Clustering/
├── data/raw/single_genre_artists.csv     # source dataset (95,837 songs)
├── src/
│   ├── config.py          # paths, features, k, sample sizes
│   ├── data_prep.py       # load, inspect, clean, log-transform, scale
│   ├── clustering.py      # k search, K-Means, DBSCAN, Agglomerative, profiling, naming
│   ├── visualize.py       # all report figures (PNG)
│   └── run_pipeline.py    # end-to-end run: python -m src.run_pipeline
├── notebooks/Amazon_Music_Clustering.ipynb   # step-by-step walkthrough (executed)
├── app/app.py             # Streamlit app (bonus)
├── outputs/amazon_music_clustered.csv        # final dataset with cluster labels
├── reports/               # CSV tables, summary.json, figures/01-14
├── models/kmeans_model.joblib                # scaler + K-Means + PCA
├── docs/Project_Report.docx, docs/screenshots/
├── run_all.bat            # Windows: install, run pipeline, execute notebook
└── push_to_github.bat     # Windows: first push to GitHub
```

## How to run

```bash
pip install -r requirements.txt
python -m src.run_pipeline                       # about 2 minutes
python -m nbconvert --to notebook --execute --inplace notebooks/Amazon_Music_Clustering.ipynb
streamlit run app/app.py
```

On Windows, double-click `run_all.bat`, then run `streamlit run app/app.py`.

## Method

1. **Exploration and cleaning:** no missing values and no exact duplicates. Removed
   1,493 re-released copies (same song, same artist, identical audio) and 84 rows with
   tempo = 0 (failed detection). 94,260 songs remain. The ID and name columns
   (`id_songs`, `name_song`, `name_artists`, which the brief calls track_id / track_name /
   artist_name) are kept for reference only.
2. **Features:** the 10 audio features from the brief. `duration_ms` is log-transformed
   (some tracks are over an hour long), then all features are standardised with
   StandardScaler.
3. **Choosing k:** K-Means for k = 2 to 10, using inertia (elbow), silhouette (on a
   fixed 20k sample), Davies-Bouldin, Calinski-Harabasz and cluster-size balance.
4. **Clustering:** final K-Means (k = 4, n_init = 20). DBSCAN (eps from the k-distance
   plot) and Agglomerative Ward are compared on the same 10k sample.
5. **Interpretation:** per-cluster feature means, z-score heatmap, rule-based names,
   top and most typical tracks, and a genre cross-tab as a check made after
   clustering (genre is never a clustering input).
6. **Visualisation:** PCA (2D) and t-SNE scatter plots, bar charts, box plots,
   heatmaps and a dendrogram.
7. **Export:** `outputs/amazon_music_clustered.csv` with `cluster`, `cluster_name`,
   `pc1` and `pc2`.

## Business use

- **Playlist curation:** each cluster is a ready-made mood pool (party/workout, chill,
  focus, audiobooks).
- **Song discovery:** the app recommends the nearest songs inside the same cluster.
- **Artist analysis:** producers can see which songs share their track's audio cluster.
- **Segmentation:** spoken-word content (13%) should be routed away from music
  recommendations.
