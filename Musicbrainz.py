import time
import csv
import musicbrainzngs
import pandas as pd
from tqdm import tqdm

# --- CONFIG ---
INPUT_CSV = "msd_metadata.csv"           # must have column 'artist_mbid'
OUTPUT_CSV = "musicbrainzdata.csv"
# Set your real app name, version and contact (email or URL)
APP_NAME = "MyMBZCollector"
APP_VERSION = "0.1"
APP_CONTACT = "olivia.robles@epfl.ch"  # required by MusicBrainz rules
REQUEST_INTERVAL = 1.05  # seconds between requests (safety margin > 1s)

# --- Initialize musicbrainzngs with user agent ---
musicbrainzngs.set_useragent(APP_NAME, APP_VERSION, APP_CONTACT)

# --- Read CSV into DataFrame ---
df = pd.read_csv(INPUT_CSV, dtype=str)
if "artist_mbid" not in df.columns:
    raise SystemExit("Input CSV must contain an 'artist_mbid' column.")

# Define main genre mapping
GENRE_MAPPING = {
    'rock': 'Rock',
    'alternative': 'Rock',
    'indie': 'Rock',
    'metal': 'Rock',
    'punk': 'Rock',
    'pop': 'Pop',
    'dance': 'Pop',
    'electronic': 'Electronic',
    'techno': 'Electronic',
    'house': 'Electronic',
    'rap': 'Hip-Hop',
    'hip-hop': 'Hip-Hop',
    'hip hop': 'Hip-Hop',
    'jazz': 'Jazz',
    'blues': 'Blues',
    'folk': 'Folk',
    'country': 'Country',
    'classical': 'Classical',
    'r&b': 'R&B',
    'soul': 'R&B',
    'reggae': 'Reggae',
    'world': 'World',
    'latin': 'Latin'
}

# Make result columns
# Ensure requested output columns exist
df["artist_mbid"] = df["artist_mbid"].astype(str)
df["name"] = None
df["gender"] = None
df["country"] = None
df["type"] = None
df["tags"] = None
df["genre_principal"] = None
df["begin_date"] = None
df["end_date"] = None
df["area"] = None
df["mbid_ok"] = True
df["mbid_status"] = None

def determine_main_genre(tags):
    """Determine the main genre from a list of tags."""
    if not tags:
        return None
    
    # Count occurrences of each main genre
    genre_counts = {}
    for tag in tags:
        tag_lower = tag.lower()
        for key, main_genre in GENRE_MAPPING.items():
            if key in tag_lower:
                genre_counts[main_genre] = genre_counts.get(main_genre, 0) + 1
    
    # Return the most frequent genre, or None if no matches
    if genre_counts:
        return max(genre_counts.items(), key=lambda x: x[1])[0]
    return None

# --- Helper to fetch artist data ---
def fetch_artist_data(mbid):
    try:
        result = musicbrainzngs.get_artist_by_id(mbid, includes=['tags'])
        artist = result.get("artist", {})

        # Basic fields
        name = artist.get('name')
        gender = artist.get('gender', None)
        country = artist.get('country', None)
        artist_type = artist.get('type', None)

        # Life-span
        life_span = artist.get('life-span', {}) or {}
        begin = life_span.get('begin', None)
        end = life_span.get('end', None)

        # Area (location)
        area = None
        if 'area' in artist and isinstance(artist['area'], dict):
            area = artist['area'].get('name')

        # Get tags
        tags = []
        if 'tag-list' in artist:
            tags = [tag.get('name') for tag in artist['tag-list'] if tag.get('name')]
        tags_str = ", ".join(tags) if tags else None

        # Determine main genre
        genre_principal = determine_main_genre(tags)

        return name, gender, country, artist_type, tags_str, genre_principal, begin, end, area, "ok"
    except musicbrainzngs.WebServiceError as e:
        return None, None, None, None, None, None, None, None, None, f"error: {e}"
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, f"error: {e}"

# --- Loop through MBIDs with rate limiting ---
for idx, row in tqdm(df.iterrows(), total=len(df)):
    mbid = str(row["artist_mbid"]).strip()
    if not mbid:
        df.at[idx, "mbid_ok"] = False
        df.at[idx, "mbid_status"] = "empty_mbid"
        continue

    name, gender, country, artist_type, tags, genre, begin, end, area, status = fetch_artist_data(mbid)
    df.at[idx, "name"] = name
    df.at[idx, "gender"] = gender
    df.at[idx, "country"] = country
    df.at[idx, "type"] = artist_type
    df.at[idx, "tags"] = tags
    df.at[idx, "genre_principal"] = genre
    df.at[idx, "begin_date"] = begin
    df.at[idx, "end_date"] = end
    df.at[idx, "area"] = area
    df.at[idx, "mbid_status"] = status

    # Respect rate limit: MusicBrainz public API ~1 request/sec
    time.sleep(REQUEST_INTERVAL)

# --- Save results ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved results to {OUTPUT_CSV}")