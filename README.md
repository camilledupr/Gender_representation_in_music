# Acoustic Differences and Artist Gender in the Million Song Dataset
**Exploring gendered patterns in timbre, pitch, and global musical descriptors**

---

## **Research Questions**
Do the acoustic characteristics of songs—timbre, pitch, key, mode, tempo, time signature, and loudness—differ systematically according to the declared social gender (male/female) of the artist?

### **Hypotheses**
- **RQ1**: Songs by male and female artists exhibit significantly different distributions in some acoustic characteristics.
- **RQ2**: Certain acoustic dimensions (for example timbre coefficients, loudness) contribute more strongly to gender differentiation.
- **RQ3**: The relationship between artist gender and acoustic features varies by genre, with differences more pronounced in specific musical contexts (for example, pop, rap).

---

## **Motivation**
Gender disparities in the music industry are well-documented:
- Only **30%** of artists in the *2022 Billboard Hot 100 Year-End Chart* were women.
- In top Billboard hits (2012–2017), **12.3%** of credited songwriters and **2.1%** of producers were women.
- Metadata-based studies reveal persistent gendered patterns in collaboration networks, career trajectories, and output, reflecting broader social biases.

This project investigates whether these inequalities extend to the **acoustic properties** of music itself.

---

## **Dataset**
A **10,000-track subset** of the *Million Song Dataset (MSD)*, enriched with *MusicBrainz* metadata:
- **Artist gender** (male/female/unknown)
- **Country, genre tags, artist type, and lifespan**
- **Acoustic features**: 12 timbre coefficients, 12 pitch-class coefficients, tempo, loudness, key, mode, and time signature.

### **Generated Datasets**
| File | Description |
|------|-------------|
| `msd_metadata.csv` | Core track metadata (tempo, key, mode, etc.) |
| `musicbrainzdata.csv` | Enriched metadata (gender, genre, country) |
| `merged.csv` | Combined dataset for analysis |
| `segments_pitches.csv` | Segment-level pitch averages |
| `segments_timbre.csv` | Segment-level timbre averages |
| `final_artist_level.csv` | Includes song-specific metadata and acoustic features |
| `final_song_level.csv` | Aggregates acoustic features and metadata at the artist level|

---

## **Project Structure**

### **1. `MSD_working.py`**
Extracts core MSD metadata and acoustic features → generates `msd_metadata.csv`.

### **2. `Musicbrainz.py`**
Fetches additional metadata (gender, genre, country) via the *MusicBrainz API* → generates `musicbrainzdata.csv`.

### **3. `merged_csv.py`**
Merges `msd_metadata.csv` and `musicbrainzdata.csv` into `merged.csv`.

### **4. `extract_segments.py`**
Computes segment-level averages for pitch and timbre → generates `segments_pitches.csv` and `segments_timbre.csv` (found in https://doi.org/10.5281/zenodo.17961667).

### **5. `representativity_subset.py`**
Assesses whether the 10,000-track subset is representative of the full MSD using statistical comparisons.

### **6. `app_visualization.py`**
Interactive *Dash* web app for exploring acoustic features by gender, genre, geography, and release year, with nearest-neighbor track recommendations.

### **7. `preprocessing.py`**
The preprocessing pipeline filters tracks by year, cleans gender and country labels, and computes mean pitch and timbre values at the song and artist levels. The resulting datasets are saved as `final_artist_level.csv` and `final_song_level.csv`.

### **8. `gender_music_analysis.ipynb`**
- **Part 1**: Data visualization (distributions, correlations).
- **Part 2**: Statistical analysis to test hypotheses (RQ1–RQ3).

---

## **How to Use**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
