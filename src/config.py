"""Project-wide settings: paths, feature list and model parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "single_genre_artists.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "songs_clean.csv"
OUTPUT_CSV = ROOT / "outputs" / "amazon_music_clustered.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

RANDOM_STATE = 42

# Audio features recommended by the brief. They describe rhythm, mood,
# instrumentation and intensity of a track.
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

# Columns kept for reference only (never used as clustering input).
# The brief calls them track_id / track_name / artist_name.
ID_COLS = ["id_songs", "name_song", "name_artists"]
REFERENCE_COLS = ID_COLS + ["genres", "release_date", "popularity_songs",
                            "popularity_artists", "followers", "explicit"]

# Range of k tried with the elbow method / silhouette score.
K_RANGE = range(2, 11)

# Final number of clusters. k=3 has the highest silhouette, but k=4 is within
# 0.012 of it, has the best (lowest) Davies-Bouldin index and separates an
# instrumental / soundtrack group that k=3 merges into the "mellow" cluster.
K_FINAL = 4

# DBSCAN: min_samples = 2 x number of features; eps read off the k-distance
# plot (about the 80th percentile). A small grid is also logged.
DBSCAN_MIN_SAMPLES = 20
DBSCAN_EPS = 1.2
DBSCAN_EPS_GRID = [0.8, 1.0, 1.2, 1.5, 1.8]

# Sample sizes for the O(n^2) steps (silhouette, DBSCAN, hierarchical, t-SNE).
SILHOUETTE_SAMPLE = 20_000
COMPARE_SAMPLE = 10_000
TSNE_SAMPLE = 6_000
DENDRO_SAMPLE = 2_000
