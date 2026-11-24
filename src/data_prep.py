"""Loading, inspection, cleaning and scaling of the song dataset."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, RAW_CSV


def load_data(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> dict:
    """Return the basic facts asked for in the brief (shape, nulls, dupes)."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_track_ids": int(df["id_songs"].duplicated().sum()),
        "repeated_song_artist_pairs": int(
            df.duplicated(["name_song", "name_artists"]).sum()),
        "zero_tempo_rows": int((df["tempo"] == 0).sum()),
        "tracks_over_10_min": int((df["duration_ms"] > 600_000).sum()),
        "unique_artists": int(df["id_artists"].nunique()),
        "unique_genres": int(df["genres"].nunique()),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the data and return it with a log of what was removed.

    1. Drop exact duplicate rows (none expected).
    2. Drop re-released copies: same song, same artist, identical audio
       features (they would only add weight to one point).
    3. Drop rows with tempo == 0 (failed tempo detection, not a real value).
    4. Add a readable genre column (the raw value is a list stored as text).
    """
    log = {"rows_in": len(df)}
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = log["rows_in"] - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["name_song", "name_artists"] + FEATURES)
    log["same_song_same_audio_removed"] = before - len(df)

    before = len(df)
    df = df[df["tempo"] > 0]
    log["zero_tempo_removed"] = before - len(df)

    df = df.copy()
    df["genre"] = (df["genres"].str.strip("[]").str.replace("'", "")
                   .str.split(",").str[0].str.strip()
                   .replace("", "unknown"))
    df = df.reset_index(drop=True)
    log["rows_out"] = len(df)
    return df, log


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the clustering features.

    duration_ms is heavily right-skewed (a few tracks run over an hour), so
    it is log-transformed before scaling to stop a handful of very long
    tracks from dominating the distance calculation.
    """
    X = df[FEATURES].copy()
    X["duration_ms"] = np.log1p(X["duration_ms"])
    return X


def scale(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardise features (mean 0, sd 1): K-Means is distance based."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler
