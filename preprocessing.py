"""
DATA PREPROCESSING FOR ACOUSTIC GENDER ANALYSIS
================================================

This script processes the Million Song Dataset subset and prepares two datasets:
1. Song-level dataset: For interactive visualizations (Dash interface, Jupyter)
2. Artist-level dataset: For statistical analyses (RQ1, RQ2, RQ3)

The preprocessing pipeline includes:
- Data cleaning and validation
- Gender and country standardization
- Segment-level acoustic feature aggregation (pitch and timbre)
- Two-level aggregation: segments -> songs -> artists

Author: Camille Dupré Tabti, Olivia Robles
Date: December 2025
Course: Foundation of Digital Humanities (DH-405), EPFL
"""

import pandas as pd
import numpy as np

# Configuration
MERGED_PATH = "merged.csv"
PITCH_PATH = "segments_pitches.csv"
TIMBRE_PATH = "segments_timbre.csv"
OUTPUT_SONG = "final_song_level.csv"
OUTPUT_ARTIST = "final_artist_level.csv"

# Load and clean metadata
df = pd.read_csv(MERGED_PATH)
print("Merged raw shape:", df.shape)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df.loc[df["year"] == 0, "year"] = np.nan
df = df[(df["year"] >= 1922) & (df["year"] <= 2011)]

for col in ["loudness", "tempo", "duration", "key", "mode", "time_signature"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["loudness", "tempo", "duration"])
print("After basic cleaning:", df.shape)

# Clean gender labels
def clean_gender(g):
    """Standardize gender labels. Test 'female' before 'male' to avoid substring issues."""
    if isinstance(g, str):
        g = g.strip()
        g_low = g.lower()
        if g_low.startswith("female"):
            return "Female"
        if g_low.startswith("male"):
            return "Male"
        if "non" in g_low and "binary" in g_low:
            return "Non-binary"
    return "Unknown"

df["gender_clean"] = df["gender"].apply(clean_gender)

# Clean country labels
df["country_clean"] = df["country"].astype(str).str.strip()
df.loc[df["country_clean"].isin(["", "nan", "None"]), "country_clean"] = "Unknown"

# Load segment-level acoustic features
print("Loading segments_pitches.csv ...")
pitch_df = pd.read_csv(PITCH_PATH)

print("Loading segments_timbre.csv ...")
timbre_df = pd.read_csv(TIMBRE_PATH)

pitch_cols = [f"pitch_{i}" for i in range(12)]
timbre_cols = [f"timbre_{i}" for i in range(12)]

for col in pitch_cols:
    pitch_df[col] = pd.to_numeric(pitch_df[col], errors="coerce")
for col in timbre_cols:
    timbre_df[col] = pd.to_numeric(timbre_df[col], errors="coerce")

# Aggregate segments to song level
group_keys_song = ["artist_mbid", "song_id"]

pitch_agg_song = (
    pitch_df
    .groupby(group_keys_song)[pitch_cols]
    .mean()
    .reset_index()
)
pitch_agg_song.columns = group_keys_song + [f"{c}_mean" for c in pitch_cols]

timbre_agg_song = (
    timbre_df
    .groupby(group_keys_song)[timbre_cols]
    .mean()
    .reset_index()
)
timbre_agg_song.columns = group_keys_song + [f"{c}_mean" for c in timbre_cols]

segments_agg_song = pitch_agg_song.merge(timbre_agg_song, on=group_keys_song, how="outer")
print("Aggregated segments to song level:", segments_agg_song.shape)

# Merge segment features into main dataframe
df = df.merge(segments_agg_song, on=group_keys_song, how="left")
print("After merging segment features:", df.shape)

# Create song-level dataset
song_level_cols = [
    "artist_mbid", "song_id", "artist_name", "title", "year",
    "gender", "gender_clean", "country", "country_clean", "genre_principal"
]

optional_acoustic = ["key", "mode", "tempo", "time_signature", "loudness", "duration"]
for col in optional_acoustic:
    if col in df.columns:
        song_level_cols.append(col)

song_level_cols += [f"pitch_{i}_mean" for i in range(12)]
song_level_cols += [f"timbre_{i}_mean" for i in range(12)]

song_level_cols = [col for col in song_level_cols if col in df.columns]
df_song = df[song_level_cols].copy()

required_song_cols = ["loudness", "tempo", "duration", "timbre_0_mean", "pitch_0_mean"]
required_song_cols = [col for col in required_song_cols if col in df_song.columns]
df_song = df_song.dropna(subset=required_song_cols)

df_song.to_csv(OUTPUT_SONG, index=False)
print(f"Saved song-level dataset: {OUTPUT_SONG} (shape: {df_song.shape})")

# Create artist-level dataset
acoustic_cols = (
    [f"pitch_{i}_mean" for i in range(12)] +
    [f"timbre_{i}_mean" for i in range(12)] +
    ["loudness", "tempo", "duration"]
)
acoustic_cols = [col for col in acoustic_cols if col in df.columns]

agg_dict = {col: "mean" for col in acoustic_cols}
metadata_cols = {
    "artist_name": "first",
    "gender": "first",
    "gender_clean": "first",
    "country_clean": "first",
    "genre_principal": "first",
}
agg_dict.update(metadata_cols)

df_artist = (
    df.groupby("artist_mbid")
    .agg(agg_dict)
    .reset_index()
)

required_artist_cols = ["loudness", "tempo", "duration", "timbre_0_mean", "pitch_0_mean"]
required_artist_cols = [col for col in required_artist_cols if col in df_artist.columns]
df_artist = df_artist.dropna(subset=required_artist_cols)

df_artist.to_csv(OUTPUT_ARTIST, index=False)
print(f"Saved artist-level dataset: {OUTPUT_ARTIST} (shape: {df_artist.shape})")

print("\nPreprocessing complete.")
print(f"Song-level: {len(df_song)} songs from {df_song['artist_mbid'].nunique()} artists")
print(f"Artist-level: {len(df_artist)} artists")
