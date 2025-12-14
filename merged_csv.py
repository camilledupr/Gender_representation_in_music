import pandas as pd

# Replace with your actual filenames
file1 = "msd_metadata.csv"
file2 = "musicbrainzdata.csv"
output_file = "merged.csv"

# Load both CSVs
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

print(f"Nombre de chansons dans msd_metadata: {len(df1)}")
print(f"Nombre de lignes dans musicbrainzdata: {len(df2)}")

# SOLUTION: Ne garder que les colonnes MusicBrainz (infos artiste) de df2
# Les colonnes qui sont spécifiques à MusicBrainz (pas dans les données de chanson)
mb_artist_columns = ['artist_mbid', 'name', 'gender', 'country', 'type', 'tags', 
                     'genre_principal', 'begin_date', 'end_date', 'area', 'mbid_ok', 'mbid_status']

# Vérifier quelles colonnes existent réellement dans df2
mb_columns_available = [col for col in mb_artist_columns if col in df2.columns]
print(f"Colonnes MusicBrainz disponibles: {mb_columns_available}")

# Garder seulement les colonnes MusicBrainz et dédupliquer par artist_mbid
df2_artists_only = df2[mb_columns_available].drop_duplicates(subset='artist_mbid', keep='first')
print(f"Nombre d'artistes uniques dans musicbrainzdata: {len(df2_artists_only)}")

# Merge on the common column "artist_mbid"
# Maintenant chaque chanson aura UNE ligne avec les infos MusicBrainz de son artiste
merged = pd.merge(df1, df2_artists_only, on="artist_mbid", how="left")

print(f"Nombre de lignes après merge: {len(merged)}")
print(f"Nombre de colonnes: {len(merged.columns)}")

# Save the merged data to a new CSV
merged.to_csv(output_file, index=False)

print(f"✅ Merged file saved as {output_file}")
print(f"✅ Une ligne par chanson ({len(merged)} chansons)")
print(f"✅ Pas de doublons ni de colonnes dupliquées")
