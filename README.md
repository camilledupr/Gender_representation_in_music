# Acoustic Differences and Artist Gender in the Million Song Dataset  
### Exploring gendered patterns in timbre, pitch, and global musical descriptors

## Research Question  
**Do the acoustic characteristics of songs: timbre, pitches, key, mode, tempo, time signature, and loudness differ systematically according to the declared social gender (male/female) of the artist?**

This project investigates whether large-scale audio features extracted from the Million Song Dataset (MSD), enriched with metadata from MusicBrainz, exhibit measurable differences between male and female artists.


## Hypotheses

### **H1 — Systematic Acoustic Differences**  
Songs by male and female artists present significantly different distributions in some acoustic characteristics.

### **H2 — Feature-Specific Effects**  
Certain acoustic dimensions (timbre coefficients or loudness...) contribute more strongly than others to gender differentiation.

### **H3 — Genre Interaction**  
The relationship between artist gender and acoustic features depends on genre.  
Differences may be more pronounced in certain musical contexts (pop, rap...) than in others.

## Motivation

### **Gender Inequalities in the Music Industry**
Gender disparities in music are widely documented across performance, authorship, and production roles.

Examples include:

- Only **30%** of artists in the *2022 Billboard Hot 100 Year-End Chart* were women.  
- In top Billboard hits (2012–2017):  
  - **12.3%** of credited songwriters were women  
  - **2.1%** of credited producers were women  

Metadata-based studies show persistent gendered patterns in collaboration networks, career trajectories, and output.

These inequalities reflect broader social processes:  
gendered stereotypes, unequal access to production tools, network segregation, and biased gatekeeping.

## Dataset Description

This project uses a **10,000-track subset** of the Million Song Dataset (MSD), enriched with MusicBrainz metadata such as:

- Artist gender *(male/female/unknown)*  
- Artist country  
- Genre tags  
- Artist type and lifespan information  

The dataset includes the following acoustic features:

- **12 timbre coefficients**  
- **12 pitch-class coefficients**  
- **tempo, loudness**  
- **key & key confidence**  
- **mode**  
- **time signature**

Additional datasets produced include:

- `msd_metadata.csv` – metadata for each track  
- `segments_pitches.csv` – segment-level chroma vectors  
- `segments_timbre.csv` – segment-level timbre vectors  
- Artist-level **12-dimensional timbre embeddings**

Although smaller than the original 1M-track MSD, this subset preserves essential pitch and timbre structure and supports meaningful exploratory audio analysis.

