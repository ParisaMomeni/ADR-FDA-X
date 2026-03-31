import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, chi2_contingency, spearmanr
from sklearn.metrics import jaccard_score

# ------------------------
# 1. Load Social Media ADR Data
# ------------------------
df = pd.read_pickle("data/1adrs2.pkl")

# Ensure Extracted_ADRs is list
df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(
    lambda x: literal_eval(str(x)) if isinstance(x, str) else x
)

# Basic stats
total_posts = len(df)
posts_with_adrs = df['Extracted_ADRs'].apply(lambda x: len(x) > 0).sum()
percentage_with_adrs = (posts_with_adrs / total_posts) * 100

print("📊 POST STATS")
print(f"Total posts: {total_posts}")
print(f"Posts with at least one ADR: {posts_with_adrs}")
print(f"Percentage with ADRs: {percentage_with_adrs:.2f}%")

# ------------------------
# 2. Flatten ADRs and Count
# ------------------------
all_adrs = [adr.lower() for adrs in df["Extracted_ADRs"] if isinstance(adrs, list) for adr in adrs]
adr_counts = Counter(all_adrs)
grouped_counts = pd.DataFrame(adr_counts.items(), columns=["ADR", "Count"]).sort_values(by="Count", ascending=False)

# Plot top 15 ADRs
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_counts.head(15), x="Count", y="ADR", hue="ADR", legend=False, palette="magma")
plt.title("Top 15 Individual ADRs in Social Media")
plt.xlabel("Mentions")
plt.ylabel("Adverse Drug Reaction")
plt.tight_layout()
plt.show()

# ------------------------
# 3. Function to Load & Normalize FDA ADR Data
# ------------------------
def load_fda_data(filepath):
    df = pd.read_excel(filepath)
    print(f"✅ Loaded: {filepath} | Columns: {df.columns.tolist()}")

    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ['ADR', 'Prevalence']
        df['ADR'] = df['ADR'].str.lower()
        df['Prevalence'] = df['Prevalence'].astype(float)
        df['FDA_norm'] = df['Prevalence'] / df['Prevalence'].sum()
        return df
    else:
        raise ValueError(f"❌ Unexpected number of columns in {filepath}")

# ------------------------
# 4. Load All FDA ADR Data
# ------------------------
fda_wegovy   = load_fda_data("data/FDAWegavy.xlsx")
fda_ozempic  = load_fda_data("data/FDAOzempic.xlsx")
fda_rybelsus = load_fda_data("data/FDARybelsus.xlsx")

# ------------------------
# 5. Merge and Compare - Wegovy Example
# ------------------------
def compare_adrs(social_df, fda_df, drug_name):
    social_df['Norm_Count'] = social_df['Count'] / social_df['Count'].sum()
    merged_df = pd.merge(social_df, fda_df[['ADR', 'FDA_norm']], on='ADR')
    merged_df.dropna(inplace=True)

    p = merged_df['Norm_Count'].values
    q = merged_df['FDA_norm'].values
    q /= q.sum()

    # KL Divergence
    kl_div = entropy(p, q)

    # Chi-Square Test
    chi2_data = np.vstack([p, q])
    chi2_stat, chi2_p, _, _ = chi2_contingency(chi2_data)

    # Jaccard Similarity
    presence_social = p > 0
    presence_fda = q > 0
    jaccard = jaccard_score(presence_social, presence_fda)

    # Spearman Correlation
    spearman_corr, spearman_p = spearmanr(p, q)

    print(f"\n📊 Comparison for {drug_name}")
    print(f"KL Divergence           : {kl_div:.4f}")
    print(f"Chi-Square Test p-value : {chi2_p:.4f}")
    print(f"Jaccard Similarity      : {jaccard:.4f}")
    print(f"Spearman Correlation    : {spearman_corr:.4f}")
    print(f"Spearman p-value        : {spearman_p:.4f}")

    # Save
    merged_df.to_csv(f"Results/ADR_Comparison_{drug_name}.csv", index=False)
    print(f"📁 Saved: ADR_Comparison_{drug_name}.csv")

# Run for all drugs
compare_adrs(grouped_counts, fda_wegovy, "Wegovy")
compare_adrs(grouped_counts, fda_ozempic, "Ozempic")
compare_adrs(grouped_counts, fda_rybelsus, "Rybelsus")
# ------------------------
def plot_fda_vs_social(merged_df, drug_name, top_n=10):
    plot_df = merged_df.sort_values(by="Count", ascending=False).head(top_n)
    plot_df = plot_df.melt(id_vars="ADR", value_vars=["Norm_Count", "FDA_norm"],
                           var_name="Source", value_name="Normalized_Frequency")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Normalized_Frequency", y="ADR", hue="Source", palette="Set2")
    plt.title(f"🔬 {drug_name} – Top {top_n} ADRs: Social Media vs FDA")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Adverse Drug Reaction")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.show()
#_ ------------------------
from math import pi

def plot_radar_chart(merged_df, drug_name, top_n=8):
    top_adrs = merged_df.sort_values(by="Count", ascending=False).head(top_n)
    categories = list(top_adrs["ADR"])
    
    values_social = list(top_adrs["Norm_Count"])
    values_fda = list(top_adrs["FDA_norm"])
    
    values_social += values_social[:1]  # close the loop
    values_fda += values_fda[:1]
    categories += categories[:1]

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    ax.plot(angles, values_social, label="Social Media", linewidth=2)
    ax.fill(angles, values_social, alpha=0.25)

    ax.plot(angles, values_fda, label="FDA", linewidth=2)
    ax.fill(angles, values_fda, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=10)
    ax.set_title(f"🕸️ Radar Chart for {drug_name} ADR Alignment", size=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
# Plot for Wegovy
#------------------------
# Load the 3 merged comparison files
weg = pd.read_csv("ADR_Comparison_Wegovy.csv")
ozem = pd.read_csv("ADR_Comparison_Ozempic.csv")
ryb = pd.read_csv("ADR_Comparison_Rybelsus.csv")

# Merge on ADR
all_adr = pd.merge(weg[['ADR', 'FDA_norm']], ozem[['ADR', 'FDA_norm']], on='ADR', suffixes=('_wegovy', '_ozempic'))
all_adr = pd.merge(all_adr, ryb[['ADR', 'FDA_norm']], on='ADR')
all_adr.rename(columns={'FDA_norm': 'FDA_norm_rybelsus'}, inplace=True)
all_adr.set_index('ADR', inplace=True)
plt.figure(figsize=(12, 8))
sns.heatmap(all_adr.T, cmap="YlOrRd", annot=True, fmt=".2f")
plt.title("🔥 FDA ADR Profile Comparison – Wegovy vs Ozempic vs Rybelsus")
plt.xlabel("ADR")
plt.ylabel("Drug")
plt.tight_layout()
plt.show()

#------------------------
# Collect metrics for each drug
stats_dict = {
    "Wegovy": {
        "KL Divergence": entropy(
            pd.read_csv("ADR_Comparison_Wegovy.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Wegovy.csv")["FDA_norm"].values
        ),
        "Spearman Correlation": spearmanr(
            pd.read_csv("ADR_Comparison_Wegovy.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Wegovy.csv")["FDA_norm"].values
        )[0]
    },
    "Ozempic": {
        "KL Divergence": entropy(
            pd.read_csv("ADR_Comparison_Ozempic.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Ozempic.csv")["FDA_norm"].values
        ),
        "Spearman Correlation": spearmanr(
            pd.read_csv("ADR_Comparison_Ozempic.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Ozempic.csv")["FDA_norm"].values
        )[0]
    },
    "Rybelsus": {
        "KL Divergence": entropy(
            pd.read_csv("ADR_Comparison_Rybelsus.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Rybelsus.csv")["FDA_norm"].values
        ),
        "Spearman Correlation": spearmanr(
            pd.read_csv("ADR_Comparison_Rybelsus.csv")["Norm_Count"].values,
            pd.read_csv("ADR_Comparison_Rybelsus.csv")["FDA_norm"].values
        )[0]
    }
}

df_stats = pd.DataFrame(stats_dict)
metrics = ["KL Divergence", "Spearman Correlation"]
subset = df_stats.loc[metrics].T

subset.plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("📈 KL Divergence and Spearman Correlation")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()
