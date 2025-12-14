import os
import sys
import glob
import time
import datetime
import csv
import h5py

# === CONFIGURATION ===
msd_subset_path = '/Users/oliviarobles/Desktop/Test FDH/MillionSongSubset copie'
msd_subset_data_path = msd_subset_path  # Data is directly in the subset folder

# === OUTPUT FILE ===
output_csv = '/Users/oliviarobles/Desktop/Test FDH/msd_metadata.csv'

# === UTILITY ===
def strtimedelta(starttime, stoptime):
    return str(datetime.timedelta(seconds=stoptime - starttime))

def apply_to_all_files(basedir, func=lambda x: x, ext='.h5'):
    cnt = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        cnt += len(files)
        for f in files:
            func(f)
    return cnt

# === EXTRACTION FUNCTION ===
rows = []

def extract_song_metadata(filename):
    """Extract metadata from a single HDF5 file"""
    try:
        with h5py.File(filename, 'r') as h5:
            song_data = []
            
            # Get the main data tables
            metadata_songs = h5['/metadata/songs'][0] if '/metadata/songs' in h5 else None
            analysis_songs = h5['/analysis/songs'][0] if '/analysis/songs' in h5 else None
            musicbrainz_songs = h5['/musicbrainz/songs'][0] if '/musicbrainz/songs' in h5 else None
            
            # Helper function to safely extract and decode data
            def safe_extract(data, field_name, default=""):
                if data is not None:
                    try:
                        value = data[field_name] if field_name in data.dtype.names else default
                        if isinstance(value, bytes):
                            return value.decode('utf-8', errors='ignore')
                        return value
                    except:
                        return default
                return default
            
            # Extract analysis data
            song_data.append(safe_extract(analysis_songs, 'analysis_sample_rate', 0.0))
            
            # Extract metadata - artist info
            song_data.append(safe_extract(metadata_songs, 'artist_7digitalid', 0))
            song_data.append(safe_extract(metadata_songs, 'artist_familiarity', 0.0))
            song_data.append(safe_extract(metadata_songs, 'artist_hotttnesss', 0.0))
            song_data.append(safe_extract(metadata_songs, 'artist_id', ''))
            song_data.append(safe_extract(metadata_songs, 'artist_latitude', 0.0))
            song_data.append(safe_extract(metadata_songs, 'artist_location', ''))
            song_data.append(safe_extract(metadata_songs, 'artist_longitude', 0.0))
            song_data.append(safe_extract(metadata_songs, 'artist_mbid', ''))
            
            # Artist tags and terms (arrays - convert to string)
            artist_mbtags = safe_extract(metadata_songs, 'artist_mbtags', [])
            song_data.append(str(artist_mbtags) if artist_mbtags else '')
            
            artist_mbtags_count = safe_extract(metadata_songs, 'artist_mbtags_count', [])
            song_data.append(str(artist_mbtags_count) if artist_mbtags_count else '')
            
            song_data.append(safe_extract(metadata_songs, 'artist_name', ''))
            song_data.append(safe_extract(metadata_songs, 'artist_playmeid', 0))
            
            artist_terms = safe_extract(metadata_songs, 'artist_terms', [])
            song_data.append(str(artist_terms) if artist_terms else '')
            
            artist_terms_freq = safe_extract(metadata_songs, 'artist_terms_freq', [])
            song_data.append(str(artist_terms_freq) if artist_terms_freq else '')
            
            artist_terms_weight = safe_extract(metadata_songs, 'artist_terms_weight', [])
            song_data.append(str(artist_terms_weight) if artist_terms_weight else '')
            
            song_data.append(safe_extract(analysis_songs, 'audio_md5', ''))
            
            # Bars data (arrays - convert to string)
            bars_confidence = safe_extract(analysis_songs, 'bars_confidence', [])
            song_data.append(str(bars_confidence) if bars_confidence else '')
            
            bars_start = safe_extract(analysis_songs, 'bars_start', [])
            song_data.append(str(bars_start) if bars_start else '')
            
            # Beats data (arrays - convert to string)
            beats_confidence = safe_extract(analysis_songs, 'beats_confidence', [])
            song_data.append(str(beats_confidence) if beats_confidence else '')
            
            beats_start = safe_extract(analysis_songs, 'beats_start', [])
            song_data.append(str(beats_start) if beats_start else '')
            
            # Song features
            song_data.append(safe_extract(analysis_songs, 'danceability', 0.0))
            song_data.append(safe_extract(analysis_songs, 'duration', 0.0))
            song_data.append(safe_extract(analysis_songs, 'end_of_fade_in', 0.0))
            song_data.append(safe_extract(analysis_songs, 'energy', 0.0))
            song_data.append(safe_extract(analysis_songs, 'key', 0))
            song_data.append(safe_extract(analysis_songs, 'key_confidence', 0.0))
            song_data.append(safe_extract(analysis_songs, 'loudness', 0.0))
            song_data.append(safe_extract(analysis_songs, 'mode', 0))
            song_data.append(safe_extract(analysis_songs, 'mode_confidence', 0.0))
            
            # Release info
            song_data.append(safe_extract(metadata_songs, 'release', ''))
            song_data.append(safe_extract(metadata_songs, 'release_7digitalid', 0))
            
            # Sections data (arrays - convert to string)
            sections_confidence = safe_extract(analysis_songs, 'sections_confidence', [])
            song_data.append(str(sections_confidence) if sections_confidence else '')
            
            sections_start = safe_extract(analysis_songs, 'sections_start', [])
            song_data.append(str(sections_start) if sections_start else '')
            
            # Segments data (arrays - convert to string)
            segments_confidence = safe_extract(analysis_songs, 'segments_confidence', [])
            song_data.append(str(segments_confidence) if segments_confidence else '')
            
            segments_loudness_max = safe_extract(analysis_songs, 'segments_loudness_max', [])
            song_data.append(str(segments_loudness_max) if segments_loudness_max else '')
            
            segments_loudness_max_time = safe_extract(analysis_songs, 'segments_loudness_max_time', [])
            song_data.append(str(segments_loudness_max_time) if segments_loudness_max_time else '')
            
            segments_loudness_start = safe_extract(analysis_songs, 'segments_loudness_start', [])
            song_data.append(str(segments_loudness_start) if segments_loudness_start else '')
            
            segments_pitches = safe_extract(analysis_songs, 'segments_pitches', [])
            song_data.append(str(segments_pitches) if segments_pitches else '')
            
            segments_start = safe_extract(analysis_songs, 'segments_start', [])
            song_data.append(str(segments_start) if segments_start else '')
            
            segments_timbre = safe_extract(analysis_songs, 'segments_timbre', [])
            song_data.append(str(segments_timbre) if segments_timbre else '')
            
            # Similar artists (array - convert to string)
            similar_artists = safe_extract(metadata_songs, 'similar_artists', [])
            song_data.append(str(similar_artists) if similar_artists else '')
            
            song_data.append(safe_extract(metadata_songs, 'song_hotttnesss', 0.0))
            song_data.append(safe_extract(metadata_songs, 'song_id', ''))
            song_data.append(safe_extract(analysis_songs, 'start_of_fade_out', 0.0))
            
            # Tatums data (arrays - convert to string)
            tatums_confidence = safe_extract(analysis_songs, 'tatums_confidence', [])
            song_data.append(str(tatums_confidence) if tatums_confidence else '')
            
            tatums_start = safe_extract(analysis_songs, 'tatums_start', [])
            song_data.append(str(tatums_start) if tatums_start else '')
            
            song_data.append(safe_extract(analysis_songs, 'tempo', 0.0))
            song_data.append(safe_extract(analysis_songs, 'time_signature', 0))
            song_data.append(safe_extract(analysis_songs, 'time_signature_confidence', 0.0))
            song_data.append(safe_extract(metadata_songs, 'title', ''))
            song_data.append(safe_extract(metadata_songs, 'track_id', ''))
            song_data.append(safe_extract(metadata_songs, 'track_7digitalid', 0))
            song_data.append(safe_extract(musicbrainz_songs, 'year', 0))
            
            rows.append(song_data)
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Add empty row to maintain file count (updated to 54 columns)
        rows.append([""] * 54)

# === MAIN EXECUTION ===
print("Extracting metadata from MSD subset using h5py...")

# Headers for the CSV - All available fields
headers = [
    "analysis_sample_rate",
    "artist_7digitalid",
    "artist_familiarity",
    "artist_hotttnesss",
    "artist_id",
    "artist_latitude",
    "artist_location",
    "artist_longitude",
    "artist_mbid",
    "artist_mbtags",
    "artist_mbtags_count",
    "artist_name",
    "artist_playmeid",
    "artist_terms",
    "artist_terms_freq",
    "artist_terms_weight",
    "audio_md5",
    "bars_confidence",
    "bars_start",
    "beats_confidence",
    "beats_start",
    "danceability",
    "duration",
    "end_of_fade_in",
    "energy",
    "key",
    "key_confidence",
    "loudness",
    "mode",
    "mode_confidence",
    "release",
    "release_7digitalid",
    "sections_confidence",
    "sections_start",
    "segments_confidence",
    "segments_loudness_max",
    "segments_loudness_max_time",
    "segments_loudness_start",
    "segments_pitches",
    "segments_start",
    "segments_timbre",
    "similar_artists",
    "song_hotttnesss",
    "song_id",
    "start_of_fade_out",
    "tatums_confidence",
    "tatums_start",
    "tempo",
    "time_signature",
    "time_signature_confidence",
    "title",
    "track_id",
    "track_7digitalid",
    "year"
]

t1 = time.time()
nfiles = apply_to_all_files(msd_subset_data_path, func=extract_song_metadata)
t2 = time.time()
print(f"Extraction completed in {strtimedelta(t1, t2)} for {nfiles} files")

# === SAVE TO CSV ===
print(f"Writing results to {output_csv} ...")
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print("✅ CSV export complete.")
print(f"Extracted data for {len(rows)} songs.")
