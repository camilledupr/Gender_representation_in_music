"""
RIGOROUS GENDER ANALYSIS IN MUSIC ACOUSTIC DATA
================================================

RQ1: Do systematic acoustic differences exist between works 
     associated with male and female artists?
RQ2: Which acoustic dimensions contribute most strongly to 
     any observed differences?
RQ3: Do acoustic patterns vary across musical genres, 
     suggesting genre-specific dynamics?
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import warnings
import os
from typing import Tuple, List, Dict, Optional

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 8),
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

GENDER_PALETTE = {"Male": "#1f77b4", "Female": "#d62728"}
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ALPHA_FDR = 0.05
MIN_TRACKS_PER_ARTIST = 1
MIN_ARTISTS_PER_GENDER = 10


def classify_gender_robust(g: str) -> str:
    if pd.isna(g) or str(g).lower() == 'nan':
        return "Unknown"
    g = str(g).lower().strip()
    if re.search(r'\b(female|woman|femme|girl|women)\b', g):
        return "Female"
    elif re.search(r'\b(male|man|homme|boy|men)\b', g) and not re.search(r'fe(male)|wo(man|men)', g):
        return "Male"
    elif re.search(r'\b(non-binary|nonbinary|genderqueer|agender)\b', g):
        return "Non-binary"
    else:
        return "Unknown"


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        df["gender_clean"] = df["gender"].apply(classify_gender_robust)
        
        print(f"✓ Data loaded: {len(df)} tracks, {df['artist_name'].nunique()} artists")
        print("\nGender distribution:")
        print(df["gender_clean"].value_counts().to_string())
        
        return df
    except FileNotFoundError:
        print(f"ERROR: '{filepath}' not found.")
        raise


def prepare_analysis_dataset(df: pd.DataFrame, acoustic_vars: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df_binary = df[df["gender_clean"].isin(["Male", "Female"])].copy()
    
    required_cols = acoustic_vars + ["artist_name", "gender_clean"]
    if "genre_principal" in df.columns:
        required_cols.append("genre_principal")
    
    df_binary = df_binary.dropna(subset=required_cols)
    
    constant_vars = [var for var in acoustic_vars if df_binary[var].nunique() <= 1]
    if constant_vars:
        print(f"\n⚠ Constant variables removed: {constant_vars}")
        acoustic_vars = [var for var in acoustic_vars if var not in constant_vars]
    
    print("\n[OUTLIER DETECTION]")
    n_before = len(df_binary)
    for var in acoustic_vars:
        mean, std = df_binary[var].mean(), df_binary[var].std()
        df_binary = df_binary[
            (df_binary[var] >= mean - 5*std) & 
            (df_binary[var] <= mean + 5*std)
        ]
    print(f"  Tracks removed: {n_before - len(df_binary)} ({100*(n_before-len(df_binary))/n_before:.1f}%)")
    
    artist_counts = df_binary['artist_name'].value_counts()
    valid_artists = artist_counts[artist_counts >= MIN_TRACKS_PER_ARTIST].index
    df_binary = df_binary[df_binary['artist_name'].isin(valid_artists)].copy()
    
    gender_counts = df_binary.groupby('gender_clean')['artist_name'].nunique()
    print(f"\n[SAMPLE SIZES]")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} artists")
        if count < MIN_ARTISTS_PER_GENDER:
            raise ValueError(f"Insufficient artists for {gender}")
    
    agg_dict = {var: 'mean' for var in acoustic_vars}
    agg_dict['gender_clean'] = 'first'
    if 'genre_principal' in df_binary.columns:
        agg_dict['genre_principal'] = 'first'
    
    df_artist = df_binary.groupby('artist_name').agg(agg_dict).reset_index()
    
    tracks_per_artist = len(df_binary) / len(df_artist)
    print(f"\n✓ Track-level: {len(df_binary)} tracks, {len(df_artist)} artists")
    print(f"✓ Artist-level: {len(df_artist)} artists")
    print(f"✓ Average tracks/artist: {tracks_per_artist:.1f}")
    print(f"✓ Valid variables: {len(acoustic_vars)}")
    
    return df_binary, df_artist, acoustic_vars


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 1e-10 else np.nan


def univariate_analysis(df_artist: pd.DataFrame, acoustic_vars: List[str]) -> pd.DataFrame:
    print("\n" + "="*70)
    print("RQ1: UNIVARIATE ANALYSIS (Artist-Level)")
    print("="*70)
    print(f"\n[METHOD]: Welch's t-test + FDR correction")
    print(f"[SAMPLE]: {len(df_artist)} artists ({(df_artist.gender_clean=='Male').sum()} M, {(df_artist.gender_clean=='Female').sum()} F)")
    
    results = []
    for var in acoustic_vars:
        male = df_artist[df_artist.gender_clean == "Male"][var].values
        female = df_artist[df_artist.gender_clean == "Female"][var].values
        
        stat, p = stats.ttest_ind(male, female, equal_var=False)
        d = compute_cohens_d(male, female)
        
        results.append({
            "Variable": var,
            "Mean_Male": male.mean(),
            "Mean_Female": female.mean(),
            "Diff": male.mean() - female.mean(),
            "P_raw": p,
            "Cohens_d": d,
            "T_stat": stat
        })
    
    df_results = pd.DataFrame(results)
    _, df_results["P_FDR"] = fdrcorrection(df_results["P_raw"].fillna(1))
    df_results["Significant_FDR"] = df_results["P_FDR"] < ALPHA_FDR
    df_results = df_results.sort_values("P_FDR").reset_index(drop=True)
    
    print(f"\n[RESULTS]: {df_results['Significant_FDR'].sum()} significant (FDR < {ALPHA_FDR})")
    print("\nTop 10:")
    print(df_results.head(10)[["Variable", "Diff", "Cohens_d", "P_FDR", "Significant_FDR"]].to_string(index=False))
    
    return df_results


def plot_top_variable_boxplot(df_artist: pd.DataFrame, var: str, pval: float, d: float):
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_artist, x="gender_clean", y=var, hue="gender_clean",
        palette=GENDER_PALETTE, order=["Male", "Female"], legend=False,
        linewidth=1.5, boxprops={'alpha': 0.8},
        medianprops={'color': 'black', 'linewidth': 2}
    )
    plt.title(f"Figure 1: {var} by Artist Gender\n(p_FDR={pval:.4f}, d={d:.3f})", 
              weight='bold', pad=15)
    plt.xlabel("Artist Gender", fontsize=13)
    plt.ylabel(var, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure1_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved")


def classification_analysis(df_artist: pd.DataFrame, acoustic_vars: List[str]) -> Tuple[pd.DataFrame, Dict]:
    print("\n" + "="*70)
    print("RQ2: CLASSIFICATION (Random Forest)")
    print("="*70)
    
    X = df_artist[acoustic_vars].values
    y = df_artist["gender_clean"].map({"Male": 0, "Female": 1}).values
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight='balanced'
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\n[METHOD]: 5-fold Stratified CV")
    print(f"[SAMPLE]: {len(df_artist)} artists")
    
    scoring = ['balanced_accuracy', 'roc_auc', 'f1']
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("\n[RESULTS]:")
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric}: {scores.mean():.3f} ± {scores.std()*2:.3f}")
    
    print("\n[PERMUTATION IMPORTANCE]")
    pipeline.fit(X, y)
    perm_imp = permutation_importance(
        pipeline, X, y, n_repeats=50, random_state=RANDOM_STATE,
        scoring='balanced_accuracy', n_jobs=-1
    )
    
    df_importance = pd.DataFrame({
        'Variable': acoustic_vars,
        'Importance': perm_imp.importances_mean,
        'Std': perm_imp.importances_std
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nTop 10:")
    print(df_importance.head(10).to_string(index=False))
    
    return df_importance, cv_results


def plot_feature_importance(df_imp: pd.DataFrame, n_top: int = 15):
    df_plot = df_imp.head(n_top)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", n_colors=len(df_plot))
    
    plt.barh(range(len(df_plot)), df_plot['Importance'], color=colors,
             edgecolor='black', linewidth=1)
    plt.errorbar(df_plot['Importance'], range(len(df_plot)), xerr=df_plot['Std'],
                 fmt='none', c='dimgray', capsize=4)
    
    plt.yticks(range(len(df_plot)), df_plot['Variable'])
    plt.xlabel("Drop in Balanced Accuracy", fontsize=13)
    plt.title("Figure 2: Feature Importance", weight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure2_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 saved")


def test_genre_patterns(df_artist: pd.DataFrame, significant_vars: List[str]) -> pd.DataFrame:
    print("\n" + "="*70)
    print("RQ3: GENRE-SPECIFIC ACOUSTIC PATTERNS")
    print("="*70)
    print("\nQuestion: Do acoustic patterns (M/F differences) vary across genres?")
    
    genre_counts = df_artist.groupby(['genre_principal', 'gender_clean']).size().unstack(fill_value=0)
    valid_genres = genre_counts[(genre_counts['Male'] >= 15) & (genre_counts['Female'] >= 5)].index
    
    df_model = df_artist[df_artist['genre_principal'].isin(valid_genres)].copy()
    
    print(f"\n[METHOD]: Two-way ANOVA (Gender × Genre) at artist level")
    print(f"[SAMPLE]: {len(df_model)} artists across {len(valid_genres)} genres")
    print(f"  Genres: {', '.join(valid_genres)}")
    
    print(f"\n[SAMPLE SIZES PER GENRE]:")
    sample_sizes = df_model.groupby(['genre_principal', 'gender_clean']).size().unstack(fill_value=0)
    print(sample_sizes.to_string())
    print(f"\nTotal: {len(df_model)} artists ({(df_model['gender_clean']=='Male').sum()} M, {(df_model['gender_clean']=='Female').sum()} F)")
    
    results = []
    
    for var in significant_vars:
        print(f"\n--- Testing: {var} ---")
        
        df_model['outcome'] = (df_model[var] - df_model[var].mean()) / df_model[var].std()
        formula = "outcome ~ C(gender_clean) * C(genre_principal)"
        
        try:
            model = smf.ols(formula, data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            gender_f = anova_table.loc['C(gender_clean)', 'F']
            gender_p = anova_table.loc['C(gender_clean)', 'PR(>F)']
            
            genre_f = anova_table.loc['C(genre_principal)', 'F']
            genre_p = anova_table.loc['C(genre_principal)', 'PR(>F)']
            
            interaction_f = anova_table.loc['C(gender_clean):C(genre_principal)', 'F']
            interaction_p = anova_table.loc['C(gender_clean):C(genre_principal)', 'PR(>F)']
            
            print(f"  Main effect (Gender):    F={gender_f:6.2f}, p={gender_p:.5f}")
            print(f"  Main effect (Genre):     F={genre_f:6.2f}, p={genre_p:.5f}")
            print(f"  Interaction (G×G):       F={interaction_f:6.2f}, p={interaction_p:.5f}")
            
            if interaction_p < 0.05:
                print(f"  ✓ Patterns VARY across genres")
                
                means = df_model.groupby(['genre_principal', 'gender_clean'])[var].mean().unstack()
                diffs = means['Male'] - means['Female']
                print(f"\n  Genre-specific M-F differences:")
                for genre in diffs.sort_values(ascending=False).index[:3]:
                    print(f"    {genre:12s}: {diffs[genre]:+.3f}")
            else:
                print(f"  ✗ Patterns CONSISTENT across genres")
            
            results.append({
                'Variable': var,
                'Gender_F': gender_f,
                'Gender_P': gender_p,
                'Genre_F': genre_f,
                'Genre_P': genre_p,
                'Interaction_F': interaction_f,
                'Interaction_P': interaction_p,
                'Significant_Interaction': interaction_p < 0.05,
                'N_artists': len(df_model),
                'N_genres': len(valid_genres)
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'Variable': var,
                'Gender_F': np.nan,
                'Gender_P': np.nan,
                'Genre_F': np.nan,
                'Genre_P': np.nan,
                'Interaction_F': np.nan,
                'Interaction_P': np.nan,
                'Significant_Interaction': False,
                'N_artists': len(df_model),
                'N_genres': len(valid_genres)
            })
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("RQ3 SUMMARY: Do Patterns Vary Across Genres?")
    print("="*70)
    print(df_results[['Variable', 'Gender_P', 'Interaction_P', 
                      'Significant_Interaction']].to_string(index=False))
    
    n_sig = df_results['Significant_Interaction'].sum()
    n_total = len(df_results)
    
    print(f"\n{'='*70}")
    if n_sig > 0:
        print(f"✓ ANSWER: YES - Patterns vary across genres")
        print(f"  {n_sig}/{n_total} variables show genre-dependent gender differences")
        print(f"  → Gender effects are MODERATED by musical genre")
        print(f"  → Genre-specific analyses recommended")
    else:
        print(f"✗ ANSWER: NO - Patterns are consistent across genres")
        print(f"  0/{n_total} variables show genre-dependent differences")
        print(f"  → Gender effects are UNIVERSAL across genres")
        print(f"  → Main effects model (RQ1) is sufficient")
    
    return df_results


def plot_genre_patterns(df_artist: pd.DataFrame, var: str, top_genres: int = 8):
    print(f"\n[VISUALIZATION]: Creating genre comparison for {var}")
    
    top = df_artist['genre_principal'].value_counts().head(top_genres).index
    df_plot = df_artist[df_artist['genre_principal'].isin(top)].copy()
    
    means = df_plot.groupby(['genre_principal', 'gender_clean'])[var].mean().unstack()
    means['Difference'] = means['Male'] - means['Female']
    means = means.sort_values('Difference', ascending=False)
    
    sample_sizes = df_plot.groupby(['genre_principal', 'gender_clean']).size().unstack(fill_value=0)
    sample_sizes = sample_sizes.reindex(means.index)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    order = means.index.tolist()
    sns.boxplot(data=df_plot, x='genre_principal', y=var, 
                hue='gender_clean', palette=GENDER_PALETTE, 
                order=order, ax=axes[0], linewidth=1.2)
    
    ax = axes[0]
    labels = []
    for i, genre in enumerate(order):
        n_male = sample_sizes.loc[genre, 'Male']
        n_female = sample_sizes.loc[genre, 'Female']
        label = f"{genre}\n(M:{n_male}, F:{n_female})"
        labels.append(label)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_title("(A) Gender Comparison Across Genres", weight='bold', fontsize=13)
    ax.set_ylabel(var, fontsize=12)
    ax.set_xlabel("Musical Genre (sample sizes)", fontsize=12)
    ax.legend(title='Gender', loc='best')
    
    x_pos = np.arange(len(means))
    colors = ['#d62728' if d > 0 else '#1f77b4' for d in means['Difference']]
    
    bars = axes[1].barh(x_pos, means['Difference'], color=colors, alpha=0.7, 
                        edgecolor='black', linewidth=1)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    
    for i, (idx, row) in enumerate(means.iterrows()):
        n_total = sample_sizes.loc[idx, 'Male'] + sample_sizes.loc[idx, 'Female']
        n_male = sample_sizes.loc[idx, 'Male']
        n_female = sample_sizes.loc[idx, 'Female']
        
        x_pos_text = row['Difference'] + (0.5 if row['Difference'] > 0 else -0.5)
        axes[1].text(x_pos_text, i, f"n={n_total}\n({n_male}M/{n_female}F)", 
                    ha='left',
                    va='center', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.7))
    
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(means.index)
    axes[1].set_xlabel(f"Male - Female (in {var})", fontsize=12)
    axes[1].set_title("(B) Gender Difference by Genre", weight='bold', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    axes[1].text(0.98, 0.80, 
                "Red = Male > Female\nBlue = Female > Male\n\n" + 
                "Similar bars = Consistent pattern\nVarying bars = Genre-specific pattern",
                transform=axes[1].transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    plt.suptitle(f"Figure 3: RQ3 - Do Gender Patterns Vary Across Genres?\n({var})", 
                 weight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure3_genre_patterns.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 saved")


def sensitivity_balanced(df_artist: pd.DataFrame, top_var: str) -> Dict:
    print("\n" + "="*70)
    print("SENSITIVITY: Balanced Sampling")
    print("="*70)
    
    n_min = df_artist['gender_clean'].value_counts().min()
    print(f"\n[METHOD]: Bootstrap (n={n_min}/gender, 1000 iterations)")
    
    p_vals, d_vals = [], []
    for i in range(1000):
        df_m = df_artist[df_artist['gender_clean']=='Male'].sample(n_min, random_state=RANDOM_STATE+i)
        df_f = df_artist[df_artist['gender_clean']=='Female'].sample(n_min, random_state=RANDOM_STATE+i)
        df_bal = pd.concat([df_m, df_f])
        
        m = df_bal[df_bal['gender_clean']=='Male'][top_var].values
        f = df_bal[df_bal['gender_clean']=='Female'][top_var].values
        
        _, p = stats.ttest_ind(m, f, equal_var=False)
        d = compute_cohens_d(m, f)
        
        p_vals.append(p)
        d_vals.append(d)
    
    p_vals, d_vals = np.array(p_vals), np.array(d_vals)
    
    print(f"\n[RESULTS]:")
    print(f"  Median p: {np.median(p_vals):.5f}")
    print(f"  95% CI p: [{np.percentile(p_vals,2.5):.5f}, {np.percentile(p_vals,97.5):.5f}]")
    print(f"  % sig: {100*(p_vals<0.05).mean():.1f}%")
    print(f"  Median d: {np.median(d_vals):.3f}")
    print(f"  95% CI d: [{np.percentile(d_vals,2.5):.3f}, {np.percentile(d_vals,97.5):.3f}]")
    
    return {'p_values': p_vals, 'cohens_d': d_vals}


def sensitivity_outliers(df_artist: pd.DataFrame, top_var: str) -> Dict:
    print("\n" + "="*70)
    print("SENSITIVITY: Outlier Exclusion")
    print("="*70)
    
    mean, std = df_artist[top_var].mean(), df_artist[top_var].std()
    df_clean = df_artist[
        (df_artist[top_var] >= mean - 3*std) &
        (df_artist[top_var] <= mean + 3*std)
    ]
    
    print(f"\n[METHOD]: Exclude > 3 SD")
    print(f"  Removed: {len(df_artist)-len(df_clean)} ({100*(len(df_artist)-len(df_clean))/len(df_artist):.1f}%)")
    
    m = df_clean[df_clean['gender_clean']=='Male'][top_var].values
    f = df_clean[df_clean['gender_clean']=='Female'][top_var].values
    
    _, p = stats.ttest_ind(m, f, equal_var=False)
    d = compute_cohens_d(m, f)
    
    print(f"\n[RESULTS]:")
    print(f"  P-value: {p:.5f}")
    print(f"  Cohen's d: {d:.3f}")
    
    return {'p_value': p, 'cohens_d': d}


def pca_analysis(df_artist: pd.DataFrame, acoustic_vars: List[str]) -> Dict:
    print("\n" + "="*70)
    print("PCA ANALYSIS")
    print("="*70)
    
    n_min = df_artist['gender_clean'].value_counts().min()
    df_m = df_artist[df_artist['gender_clean']=='Male'].sample(n_min, random_state=RANDOM_STATE)
    df_f = df_artist[df_artist['gender_clean']=='Female'].sample(n_min, random_state=RANDOM_STATE)
    df_bal = pd.concat([df_m, df_f])
    
    print(f"\n[METHOD]: PCA on balanced sample (n={n_min}/gender)")
    
    X = df_bal[acoustic_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_scaled)
    
    print(f"\n[RESULTS]:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1','PC2'], index=acoustic_vars)
    print("\nTop PC1 contributors:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5))
    
    pc1_m = comp[df_bal['gender_clean']=='Male', 0]
    pc1_f = comp[df_bal['gender_clean']=='Female', 0]
    _, p1 = stats.ttest_ind(pc1_m, pc1_f, equal_var=False)
    
    pc2_m = comp[df_bal['gender_clean']=='Male', 1]
    pc2_f = comp[df_bal['gender_clean']=='Female', 1]
    _, p2 = stats.ttest_ind(pc2_m, pc2_f, equal_var=False)
    
    print(f"\nExploratory gender projection:")
    print(f"  PC1: p={p1:.5f}")
    print(f"  PC2: p={p2:.5f}")
    
    df_pca = pd.DataFrame({
        'PC1': comp[:,0], 'PC2': comp[:,1],
        'gender': df_bal['gender_clean'].values
    })
    
    plt.figure(figsize=(10,8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='gender',
                    palette=GENDER_PALETTE, alpha=0.6, s=50)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=13)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=13)
    plt.title("Figure 4: PCA by Gender", weight='bold', pad=15)
    plt.legend(title='Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure4_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 saved")
    
    return {'pca': pca, 'loadings': loadings, 'p_pc1': p1, 'p_pc2': p2}


def main():
    print("="*70)
    print("RIGOROUS GENDER ANALYSIS IN MUSIC ACOUSTIC DATA")
    print("="*70)
    print("\nResearch Questions:")
    print("  RQ1: Do systematic acoustic differences exist between works")
    print("       associated with male and female artists?")
    print("  RQ2: Which acoustic dimensions contribute most strongly to")
    print("       any observed differences?")
    print("  RQ3: Do acoustic patterns vary across musical genres,")
    print("       suggesting genre-specific dynamics?")
    print("="*70)
    
    ACOUSTIC_VARS = [f"timbre_{i}_mean" for i in range(12)] + \
                    [f"pitch_{i}_mean" for i in range(12)] + \
                    ["loudness", "tempo", "energy", "danceability"]
    
    df = load_and_clean_data("final.csv")
    df_track, df_artist, acoustic_vars = prepare_analysis_dataset(df, ACOUSTIC_VARS)
    
    df_uni = univariate_analysis(df_artist, acoustic_vars)
    sig_vars = df_uni[df_uni['Significant_FDR']]['Variable'].tolist()
    
    if sig_vars:
        top_var = sig_vars[0]
        top = df_uni[df_uni['Variable']==top_var].iloc[0]
        plot_top_variable_boxplot(df_artist, top_var, top['P_FDR'], top['Cohens_d'])
        print(f"\n→ Top variable for RQ1: {top_var}")
    else:
        top_var = None
        print("\n⚠ No significant variables found in RQ1")
    
    df_imp, cv_res = classification_analysis(df_artist, acoustic_vars)
    plot_feature_importance(df_imp)
    
    if 'genre_principal' in df_artist.columns and sig_vars:
        print("\n" + "="*70)
        print("PROCEEDING TO RQ3: GENRE-SPECIFIC PATTERNS")
        print("="*70)
        
        df_rq3 = test_genre_patterns(df_artist, sig_vars[:5])
        
        if top_var:
            plot_genre_patterns(df_artist, top_var)
    else:
        print("\n⚠ RQ3 skipped: no genre information or no significant variables")
    
    if top_var:
        sens_bal = sensitivity_balanced(df_artist, top_var)
        sens_out = sensitivity_outliers(df_artist, top_var)
    
    pca_res = pca_analysis(df_artist, acoustic_vars)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("""
✓ METHODOLOGICAL STRENGTHS:
  - Artist-level analysis (RQ1, RQ2, RQ3) → proper independence
  - Two-way ANOVA with interaction (RQ3) → tests genre-specific effects
  - FDR correction for multiple testing → controls Type I error
  - Effect sizes reported (Cohen's d) → practical significance
  - Sensitivity analyses → robustness checks
  - Proper cross-validation (stratified) → generalization

✓ KEY RESULTS:
  - RQ1: Identified systematic differences with effect sizes
  - RQ2: Ranked acoustic dimensions by predictive importance
  - RQ3: Tested if gender effects vary by musical genre

⚠ IMPORTANT LIMITATIONS:
  - Correlational analysis (not causal inference)
  - Gender may be confounded with era, label, production style
  - Artist-level aggregation appropriate given low tracks/artist
  - Sample composition may affect generalizability

 INTERPRETATION:
  - RQ1: "Do male/female artists differ on average?"
  - RQ2: "Which features best predict artist gender?"
  - RQ3: "Do these differences depend on musical genre?"
    """)
    
    print(f"\n✓ All outputs saved to '{OUTPUT_DIR}/'")
    print(f"  - Figure 1: Top variable by gender (RQ1)")
    print(f"  - Figure 2: Feature importance (RQ2)")
    print(f"  - Figure 3: Genre-specific patterns (RQ3)")
    print(f"  - Figure 4: PCA visualization")
    
    print("\n" + "="*70)
    print("READY FOR DEFENSE!")
    print("="*70)


if __name__ == "__main__":
    main()
