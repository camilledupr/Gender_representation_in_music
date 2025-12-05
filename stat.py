import pandas as pd
import numpy as np

# Load metadata
meta = pd.read_csv("merged.csv")

# Load per-segment timbre
timbre = pd.read_csv("segments_timbre.csv")

# Compute an aggregate timbre for each artist (mean over segments)
timbre_cols = [f'timbre_{i}' for i in range(12)]
artist_timbre = timbre.groupby('artist_mbid')[timbre_cols].mean().reset_index()

# Merge with metadata
data = meta.merge(artist_timbre, on='artist_mbid')

# Filter out rows with missing genre or gender data
data_clean = data.dropna(subset=['genre_principal', 'gender'])
print(f"Data after filtering: {len(data_clean)} samples")

# Timbre embeddings
emb = data_clean[timbre_cols].values

# Labels
genres = data_clean['genre_principal'].values
artist_gender = data_clean['gender'].values

from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

genres_unique = np.unique(genres)
dist = cdist(emb, emb, metric='euclidean')

intra_dist = []
inter_dist = []

for g in genres_unique:
    idx = np.where(genres == g)[0]
    others = np.where(genres != g)[0]
    d_intra = dist[np.ix_(idx, idx)]
    intra_dist.extend(d_intra[np.triu_indices(len(idx), k=1)])
    d_inter = dist[np.ix_(idx, others)]
    inter_dist.extend(d_inter.flatten())

t_stat, p_val = ttest_ind(intra_dist, inter_dist, equal_var=False)
print("T-test intra vs inter genres p-value:", p_val)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

le = LabelEncoder()
labels = le.fit_transform(genres)

sil = silhouette_score(emb, labels)
print("Silhouette score genres =", sil)

from statsmodels.multivariate.manova import MANOVA

df = pd.DataFrame(emb, columns=timbre_cols)
df['gender'] = artist_gender
df['genre'] = genres

man = MANOVA.from_formula('timbre_0 + timbre_1 + timbre_2 + timbre_3 + timbre_4 + timbre_5 + timbre_6 + timbre_7 + timbre_8 + timbre_9 + timbre_10 + timbre_11 ~ gender + genre', data=df)
print(man.mv_test())

from scipy.spatial.distance import cdist

def permutation_distance_test(X1, X2, n_perm=1000):
    observed = np.mean(cdist(X1, X2))
    combined = np.vstack([X1, X2])
    n1 = len(X1)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(combined))
        X1_perm = combined[perm[:n1]]
        X2_perm = combined[perm[n1:]]
        if np.mean(cdist(X1_perm, X2_perm)) <= observed:
            count += 1
    return count / n_perm

results = {}
for g in genres_unique:
    idx = np.where(genres == g)[0]
    X = emb[idx]
    gender_g = artist_gender[idx]
    X_m = X[gender_g == 'male']
    X_f = X[gender_g == 'female']
    if len(X_m) > 5 and len(X_f) > 5:
        p = permutation_distance_test(X_m, X_f)
        results[g] = p

print("Permutation test p-values per genre:", results)

# ====================================
# ANALYSES SUPPLEMENTAIRES POUR Q2 : GENRE SOCIAL ET TIMBRE
# ====================================

print("\n" + "="*60)
print("QUESTION 2: TIMBRE ET GENRE SOCIAL DE L'ARTISTE")
print("="*60)

# Analyse globale des différences homme/femme (sans contrôle du genre musical)
from scipy.stats import mannwhitneyu

print(f"\n📊 Distribution par genre social:")
gender_counts = pd.Series(artist_gender).value_counts()
print(gender_counts)

# Test global des différences timbrales homme/femme
male_emb = emb[artist_gender == 'Male']
female_emb = emb[artist_gender == 'Female']

print(f"\nÉchantillons: {len(male_emb)} hommes, {len(female_emb)} femmes")

# Test de Mann-Whitney pour chaque dimension timbrale
print(f"\n🧪 Tests Mann-Whitney par dimension timbrale (H vs F):")
timbre_pvals = []
for i, col in enumerate(timbre_cols):
    male_vals = male_emb[:, i]
    female_vals = female_emb[:, i]
    stat, pval = mannwhitneyu(male_vals, female_vals, alternative='two-sided')
    timbre_pvals.append(pval)
    significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {col}: p = {pval:.6f} {significance}")

# Correction de Bonferroni
corrected_pvals = np.array(timbre_pvals) * len(timbre_pvals)
n_significant = sum(p < 0.05 for p in corrected_pvals)
print(f"\nAprès correction Bonferroni: {n_significant}/{len(timbre_cols)} dimensions significatives")

# Analyse par les genres musicaux les plus représentés
print(f"\n🎵 Analyse par genre musical (genres avec >50 échantillons):")
genre_analysis = {}
for genre in ['Pop', 'Rock', 'Jazz', 'Blues', 'Hip-Hop', 'Country']:
    genre_mask = genres == genre
    if genre_mask.sum() > 50:  # Seuil minimal
        genre_data = data_clean[genre_mask]
        male_count = (genre_data['gender'] == 'Male').sum()
        female_count = (genre_data['gender'] == 'Female').sum()
        
        if male_count >= 10 and female_count >= 10:
            genre_male_emb = emb[genre_mask & (artist_gender == 'Male')]
            genre_female_emb = emb[genre_mask & (artist_gender == 'Female')]
            
            # Test multivarié sur ce genre spécifique
            from scipy.stats import energy_distance
            energy_dist = energy_distance(genre_male_emb.flatten(), 
                                        genre_female_emb.flatten())
            
            genre_analysis[genre] = {
                'male_n': male_count,
                'female_n': female_count,
                'energy_distance': energy_dist
            }
            
            print(f"  {genre}: {male_count}M/{female_count}F, distance énergétique = {energy_dist:.4f}")

# Effet d'interaction genre musical × genre social
print(f"\n🔄 Test d'interaction genre musical × genre social:")
# Utiliser un modèle linéaire pour tester l'interaction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Encoder les variables catégorielles
genre_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
genre_encoded = genre_encoder.fit_transform(genres)
gender_encoded = gender_encoder.fit_transform(artist_gender)

# Créer terme d'interaction
interaction_term = genre_encoded * gender_encoded

# Test pour la première dimension timbrale comme exemple
X = np.column_stack([genre_encoded, gender_encoded, interaction_term])
y = emb[:, 0]  # Première dimension timbrale

model = LinearRegression().fit(X, y)
r2 = model.score(X, y)
print(f"  R² du modèle (timbre_0 ~ genre + gender + interaction): {r2:.4f}")

# Calculer la contribution de l'interaction
X_no_interaction = X[:, :2]  # Sans interaction
model_no_int = LinearRegression().fit(X_no_interaction, y)
r2_no_int = model_no_int.score(X_no_interaction, y)
interaction_contribution = r2 - r2_no_int
print(f"  Contribution de l'interaction: ΔR² = {interaction_contribution:.6f}")
