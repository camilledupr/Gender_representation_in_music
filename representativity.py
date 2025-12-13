import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


subset_color = "#FF007F"  
full_color = "#004ACB"   

sns.set_style("whitegrid")

# =========================
# LOAD SUBSET DATASET
# =========================

df = pd.read_csv("final.csv")

# --- Genre ---
if "genre_principal" not in df.columns:
    raise ValueError("La colonne 'genre_principal' n'existe pas dans final.csv")

subset_genre_df = (
    df["genre_principal"]
    .dropna()
    .value_counts()
    .to_frame(name="Count")
)

subset_genre_df["Percentage"] = (
    subset_genre_df["Count"] / subset_genre_df["Count"].sum() * 100
)

# --- Gender ---
if "gender" not in df.columns:
    raise ValueError("La colonne 'gender' n'existe pas dans final.csv")

subset_gender_df = (
    df["gender"]
    .fillna("Unknown")
    .value_counts()
    .to_frame(name="Count")
)

subset_gender_df["Percentage"] = (
    subset_gender_df["Count"] / subset_gender_df["Count"].sum() * 100
)

# --- Country ---
if "country" not in df.columns:
    raise ValueError("La colonne 'country' n'existe pas dans final.csv")

subset_country_df = (
    df["country"]
    .dropna()
    .value_counts()
    .head(10)  # Top 10 countries
    .to_frame(name="Count")
)

subset_country_df["Percentage"] = (
    subset_country_df["Count"] / subset_country_df["Count"].sum() * 100
)

# =========================
# FULL DATASET (UNCHANGED)
# =========================

full_genre_distribution = {
    "Pop_Rock": 37120, "Rock": 21831, "Pop": 5180, "Metal": 4661,
    "Country": 4201, "Rap": 4174, "RnB": 3803, "Electronic": 3601,
    "Latin": 2091, "Folk": 1920, "Reggae": 1537, "Jazz": 1296,
    "Punk": 1261, "Blues": 862, "World": 433, "International": 249,
    "New Age": 194, "Vocal": 68
}

full_gender_distribution = {
    "Male": 900971,
    "Female": 269220,
    "Non-binary": 1949,
    "Unknown": 942576 + 2250 + 1646
}

full_country_distribution = {
    "US": 332835,
    "GB": 125073,  # UK -> GB
    "JP": 106583,  # Japan -> JP
    "DE": 103033,  # Germany -> DE
    "FR": 60747,   # France -> FR
    "BE": 45163,   # Belgium -> BE
    "CA": 44171,   # Canada -> CA
    "IT": 40700,   # Italy -> IT
    "AU": 33232,   # Australia -> AU
    "NL": 29384    # The Netherlands -> NL
}

full_genre_df = pd.DataFrame.from_dict(
    full_genre_distribution, orient="index", columns=["Count"]
)
full_genre_df["Percentage"] = (
    full_genre_df["Count"] / full_genre_df["Count"].sum() * 100
)

full_gender_df = pd.DataFrame.from_dict(
    full_gender_distribution, orient="index", columns=["Count"]
)
full_gender_df["Percentage"] = (
    full_gender_df["Count"] / full_gender_df["Count"].sum() * 100
)

full_country_df = pd.DataFrame.from_dict(
    full_country_distribution, orient="index", columns=["Count"]
)
full_country_df["Percentage"] = (
    full_country_df["Count"] / full_country_df["Count"].sum() * 100
)

# =========================
# PITCH & TIMBRE
# =========================

subset_pitch_stats = {
    "pitch_0": {"Mean": 0.448127, "Std Dev": 0.147934},
    "pitch_1": {"Mean": 0.443889, "Std Dev": 0.149909},
    "pitch_2": {"Mean": 0.365789, "Std Dev": 0.116380},
    "pitch_3": {"Mean": 0.308670, "Std Dev": 0.110634},
    "pitch_4": {"Mean": 0.356575, "Std Dev": 0.120478},
    "pitch_5": {"Mean": 0.328588, "Std Dev": 0.112492},
    "pitch_6": {"Mean": 0.335250, "Std Dev": 0.112200},
    "pitch_7": {"Mean": 0.363492, "Std Dev": 0.120032},
    "pitch_8": {"Mean": 0.329858, "Std Dev": 0.114865},
    "pitch_9": {"Mean": 0.362058, "Std Dev": 0.121390},
    "pitch_10": {"Mean": 0.310694, "Std Dev": 0.108811},
    "pitch_11": {"Mean": 0.337013, "Std Dev": 0.114209}
}

subset_timbre_stats = {
    "timbre_0": {"Mean": 43.376, "Std Dev": 6.209},
    "timbre_1": {"Mean": 3.861, "Std Dev": 50.911},
    "timbre_2": {"Mean": 11.575, "Std Dev": 34.113},
    "timbre_3": {"Mean": 1.257, "Std Dev": 15.421},
    "timbre_4": {"Mean": -6.223, "Std Dev": 22.488},
    "timbre_5": {"Mean": -7.883, "Std Dev": 13.908},
    "timbre_6": {"Mean": -3.074, "Std Dev": 14.447},
    "timbre_7": {"Mean": -1.446, "Std Dev": 7.812},
    "timbre_8": {"Mean": 3.506, "Std Dev": 10.677},
    "timbre_9": {"Mean": 2.181, "Std Dev": 6.546},
    "timbre_10": {"Mean": -0.565, "Std Dev": 4.349},
    "timbre_11": {"Mean": 2.684, "Std Dev": 8.195}
}

full_pitch_stats = {
    "pitch_0": {"Mean": 0.45, "Std Dev": 0.15},
    "pitch_1": {"Mean": 0.45, "Std Dev": 0.15},
    "pitch_2": {"Mean": 0.37, "Std Dev": 0.12},
    "pitch_3": {"Mean": 0.31, "Std Dev": 0.11},
    "pitch_4": {"Mean": 0.37, "Std Dev": 0.12},
    "pitch_5": {"Mean": 0.33, "Std Dev": 0.11},
    "pitch_6": {"Mean": 0.34, "Std Dev": 0.11},
    "pitch_7": {"Mean": 0.36, "Std Dev": 0.12},
    "pitch_8": {"Mean": 0.33, "Std Dev": 0.11},
    "pitch_9": {"Mean": 0.36, "Std Dev": 0.12},
    "pitch_10": {"Mean": 0.30, "Std Dev": 0.11},
    "pitch_11": {"Mean": 0.35, "Std Dev": 0.12}
}

full_timbre_stats = {
    "timbre_0": {"Mean": 45.34, "Std Dev": 5.16},
    "timbre_1": {"Mean": 10.27, "Std Dev": 43.04},
    "timbre_2": {"Mean": 10.47, "Std Dev": 27.61},
    "timbre_3": {"Mean": -2.48, "Std Dev": 11.87},
    "timbre_4": {"Mean": -9.96, "Std Dev": 20.82},
    "timbre_5": {"Mean": -11.89, "Std Dev": 10.86},
    "timbre_6": {"Mean": -2.06, "Std Dev": 12.07},
    "timbre_7": {"Mean": -2.50, "Std Dev": 6.63},
    "timbre_8": {"Mean": 4.47, "Std Dev": 8.92},
    "timbre_9": {"Mean": 1.65, "Std Dev": 6.09},
    "timbre_10": {"Mean": -0.41, "Std Dev": 3.54},
    "timbre_11": {"Mean": 1.93, "Std Dev": 7.33}
}

subset_pitch_df = pd.DataFrame(subset_pitch_stats).T
subset_timbre_df = pd.DataFrame(subset_timbre_stats).T
full_pitch_df = pd.DataFrame(full_pitch_stats).T
full_timbre_df = pd.DataFrame(full_timbre_stats).T

# =========================
# PLOTTING
# =========================

# --- Genre ---
plt.figure(figsize=(12, 6))
sns.barplot(
    x=subset_genre_df.index,
    y="Percentage",
    data=subset_genre_df,
    label="Subset",
    color=subset_color
)
sns.barplot(
    x=full_genre_df.index,
    y="Percentage",
    data=full_genre_df,
    alpha=0.6,
    label="Full Dataset",
    color=full_color
)
plt.xticks(rotation=90)
plt.title("Genre Distribution Comparison", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("genre_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Gender ---
plt.figure(figsize=(10, 6))
sns.barplot(
    x=subset_gender_df.index,
    y="Percentage",
    data=subset_gender_df,
    label="Subset",
    color=subset_color
)
sns.barplot(
    x=full_gender_df.index,
    y="Percentage",
    data=full_gender_df,
    alpha=0.6,
    label="Full Dataset",
    color=full_color
)
plt.xticks(rotation=45)
plt.title("Gender Distribution Comparison", fontsize=16)
plt.ylabel("Percentage (%)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("gender_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Country ---
plt.figure(figsize=(14, 8))

# Create a combined plot with subset and full dataset
# Get top 10 from both datasets for comparison
subset_top10 = subset_country_df.head(10)
full_top10 = full_country_df.head(10)

# Create positions for bars
x_positions = range(len(subset_top10))
width = 0.35

# Plot subset data
plt.bar([x - width/2 for x in x_positions], 
        subset_top10["Percentage"], 
        width, 
        label="Subset", 
        color=subset_color,
        alpha=0.8)

# For full dataset, we need to match countries or show them separately
# Let's create a comprehensive comparison
all_countries = set(subset_top10.index) | set(full_top10.index)

# Create aligned data
subset_aligned = []
full_aligned = []
country_labels = []

for country in subset_top10.index:
    country_labels.append(country)
    subset_aligned.append(subset_top10.loc[country, "Percentage"])
    if country in full_country_df.index:
        full_aligned.append(full_country_df.loc[country, "Percentage"])
    else:
        full_aligned.append(0)

# Plot full dataset data
plt.bar([x + width/2 for x in range(len(country_labels))], 
        full_aligned, 
        width, 
        label="Full Dataset", 
        color=full_color,
        alpha=0.6)

plt.xlabel('Country', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.title('Top 10 Countries Distribution Comparison', fontsize=16)
plt.xticks(range(len(country_labels)), country_labels, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("country_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# Print comparison statistics
print("\n" + "="*60)
print("COUNTRY DISTRIBUTION COMPARISON")
print("="*60)
print(f"{'Country':<15} {'Subset %':<12} {'Full Dataset %':<15} {'Difference':<12}")
print("-" * 60)
for country in subset_top10.index:
    subset_pct = subset_top10.loc[country, "Percentage"]
    full_pct = full_country_df.loc[country, "Percentage"] if country in full_country_df.index else 0
    difference = subset_pct - full_pct
    print(f"{country:<15} {subset_pct:<12.2f} {full_pct:<15.2f} {difference:<12.2f}")
print("="*60)

# --- Pitch ---
plt.figure(figsize=(12, 6))
pitch_comparison = pd.concat(
    [subset_pitch_df, full_pitch_df],
    axis=1,
    keys=["Subset", "Full Dataset"]
)
sns.lineplot(
    data=pitch_comparison.xs("Mean", level=1, axis=1)["Subset"],
    marker="o",
    color=subset_color,
    label="Subset",
    linewidth=2.5
)
sns.lineplot(
    data=pitch_comparison.xs("Mean", level=1, axis=1)["Full Dataset"],
    marker="o",
    color=full_color,
    label="Full Dataset",
    linewidth=2.5
)
plt.title("Pitch Mean Comparison", fontsize=16)
plt.ylabel("Mean", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("pitch_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Timbre ---
plt.figure(figsize=(12, 6))
timbre_comparison = pd.concat(
    [subset_timbre_df, full_timbre_df],
    axis=1,
    keys=["Subset", "Full Dataset"]
)
sns.lineplot(
    data=timbre_comparison.xs("Mean", level=1, axis=1)["Subset"],
    marker="o",
    color=subset_color,
    label="Subset",
    linewidth=2.5
)
sns.lineplot(
    data=timbre_comparison.xs("Mean", level=1, axis=1)["Full Dataset"],
    marker="o",
    color=full_color,
    label="Full Dataset",
    linewidth=2.5
)
plt.title("Timbre Mean Comparison", fontsize=16)
plt.ylabel("Mean", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("timbre_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

