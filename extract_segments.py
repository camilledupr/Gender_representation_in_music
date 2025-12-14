import os
import sys
import glob
import time
import datetime
import csv
import h5py
import numpy as np

# === CONFIGURATION ===
msd_subset_path = '/Users/oliviarobles/Desktop/Test FDH/MillionSongSubset copie'
msd_subset_data_path = msd_subset_path  # Data is directly in the subset folder

# === OUTPUT FILES ===
output_pitches_csv = '/Users/oliviarobles/Desktop/Test FDH/segments_pitches.csv'
output_timbre_csv = '/Users/oliviarobles/Desktop/Test FDH/segments_timbre.csv'

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
pitches_data = []
timbre_data = []

def extract_segments_data(filename):
    """Extract segments pitches and timbre from a single HDF5 file"""
    try:
        with h5py.File(filename, 'r') as h5:
            # Get the main data tables
            metadata_songs = h5['/metadata/songs'][0] if '/metadata/songs' in h5 else None
            analysis_songs = h5['/analysis/songs'][0] if '/analysis/songs' in h5 else None
            
            # Extract artist_mbid
            artist_mbid = ''
            if metadata_songs is not None and 'artist_mbid' in metadata_songs.dtype.names:
                mbid = metadata_songs['artist_mbid']
                if isinstance(mbid, bytes):
                    artist_mbid = mbid.decode('utf-8', errors='ignore')
                else:
                    artist_mbid = str(mbid)
            
            # Extract song_id for reference
            song_id = ''
            if metadata_songs is not None and 'song_id' in metadata_songs.dtype.names:
                sid = metadata_songs['song_id']
                if isinstance(sid, bytes):
                    song_id = sid.decode('utf-8', errors='ignore')
                else:
                    song_id = str(sid)
            
            # Extract track_id
            track_id = ''
            if metadata_songs is not None and 'track_id' in metadata_songs.dtype.names:
                tid = metadata_songs['track_id']
                if isinstance(tid, bytes):
                    track_id = tid.decode('utf-8', errors='ignore')
                else:
                    track_id = str(tid)
            
            # Extract title
            title = ''
            if metadata_songs is not None and 'title' in metadata_songs.dtype.names:
                t = metadata_songs['title']
                if isinstance(t, bytes):
                    title = t.decode('utf-8', errors='ignore')
                else:
                    title = str(t)
            
            # Extract artist_name
            artist_name = ''
            if metadata_songs is not None and 'artist_name' in metadata_songs.dtype.names:
                an = metadata_songs['artist_name']
                if isinstance(an, bytes):
                    artist_name = an.decode('utf-8', errors='ignore')
                else:
                    artist_name = str(an)
            
            # Extract segments_pitches (2D array: n_segments x 12)
            if 'segments_pitches' in h5['/analysis']:
                pitches = h5['/analysis/segments_pitches'][:]
                # Each row is one segment with 12 pitch values
                for segment_idx, pitch_vector in enumerate(pitches):
                    row = [artist_mbid, song_id, track_id, title, artist_name, segment_idx]
                    row.extend(pitch_vector.tolist())
                    pitches_data.append(row)
            
            # Extract segments_timbre (2D array: n_segments x 12)
            if 'segments_timbre' in h5['/analysis']:
                timbre = h5['/analysis/segments_timbre'][:]
                # Each row is one segment with 12 timbre values
                for segment_idx, timbre_vector in enumerate(timbre):
                    row = [artist_mbid, song_id, track_id, title, artist_name, segment_idx]
                    row.extend(timbre_vector.tolist())
                    timbre_data.append(row)
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# === MAIN EXECUTION ===
print("Extracting segments pitches and timbre from MSD subset...")
print("This will create one row per segment (each song has multiple segments)")

t1 = time.time()
nfiles = apply_to_all_files(msd_subset_data_path, func=extract_segments_data)
t2 = time.time()
print(f"Extraction completed in {strtimedelta(t1, t2)} for {nfiles} files")

# === SAVE PITCHES TO CSV ===
print(f"\nWriting segments_pitches to {output_pitches_csv} ...")
pitches_headers = ['artist_mbid', 'song_id', 'track_id', 'title', 'artist_name', 'segment_idx'] + \
                  [f'pitch_{i}' for i in range(12)]

with open(output_pitches_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(pitches_headers)
    writer.writerows(pitches_data)

print(f"✅ Segments pitches CSV complete: {len(pitches_data)} segment rows")

# === SAVE TIMBRE TO CSV ===
print(f"\nWriting segments_timbre to {output_timbre_csv} ...")
timbre_headers = ['artist_mbid', 'song_id', 'track_id', 'title', 'artist_name', 'segment_idx'] + \
                 [f'timbre_{i}' for i in range(12)]

with open(output_timbre_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(timbre_headers)
    writer.writerows(timbre_data)

print(f" Segments timbre CSV complete: {len(timbre_data)} segment rows")

print("\n" + "="*60)
print("SUMMARY:")
print(f"- Processed {nfiles} songs")
print(f"- Extracted {len(pitches_data)} pitch segments")
print(f"- Extracted {len(timbre_data)} timbre segments")
print(f"- Each segment has 12 features (pitch_0 to pitch_11 / timbre_0 to timbre_11)")
print(f"- Files created:")
print(f"  • {output_pitches_csv}")
print(f"  • {output_timbre_csv}")
print("="*60)
