import pandas as pd
from collections import Counter
import scipy.stats as stats

# -------------------------------
# Load Twitter ADR data
# -------------------------------
df_tw = pd.read_pickle("Results/1adrs2_with_extractions.pkl")
df_tw["Extracted_ADRs"] = df_tw["Extracted_ADRs"].apply(
    lambda x: [i.lower() for i in x] if isinstance(x, list) else []
)
df_tw["Brand"] = df_tw["Primary_Brand"].fillna('')

# -------------------------------
# Load and normalize FDA data
# -------------------------------
fda_info = {
    "Wegovy":      ("data/FDAWegavy.xlsx", "FDA 2.4 mg"),
    "Ozempic_0.5": ("data/FDAOzempic.xlsx", "FDA 0.5 mg"),
    "Ozempic_1":   ("data/FDAOzempic.xlsx", "FDA 1 mg"),
    "Rybelsus_7":  ("data/FDARybelsus.xlsx", "FDA 7 mg"),
    "Rybelsus_14": ("data/FDARybelsus.xlsx", "FDA 14 mg"),
}

def load_fda(path, col):
    dfx = pd.read_excel(path)
    dfx = dfx.rename(columns={dfx.columns[0]: "ADR", col: "Prevalence"})
    dfx = dfx[["ADR", "Prevalence"]].dropna()
    dfx["ADR"] = dfx["ADR"].str.lower()
    dfx["Prevalence"] = dfx["Prevalence"].astype(str).str.replace("%", "", regex=False).astype(float)
    #Step 1: Normalize FDA percentages
    tot = dfx["Prevalence"].sum()
    return dict(zip(dfx["ADR"], dfx["Prevalence"]/tot)) if tot else {}

# Aggregate FDA values to brand level
brand_level_fda = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    keys = [k for k in fda_info if k.startswith(brand)]
    acc = Counter()
    for k in keys:
        acc.update(load_fda(*fda_info[k]))
    tot = sum(acc.values())
    brand_level_fda[brand] = {k: v/tot for k, v in acc.items()} if tot else {}

# -------------------------------
# Aggregate Twitter ADRs per brand
# -------------------------------
twitter_adr_counts = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
    adrs = [
        adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"]
        for adr in row
    ]
    cnt = Counter(adrs)
    tot = sum(cnt.values())
    twitter_adr_counts[brand] = {k: v/tot for k, v in cnt.items()} if tot else {}

# -------------------------------
# Identify top 10 ADRs across all sources
# -------------------------------
all_counters = list(twitter_adr_counts.values()) + [load_fda(*v) for v in fda_info.values()]
combined = Counter()
for c in all_counters:
    combined.update(c)
top_adrs = [a for a, _ in combined.most_common(10)]
print("Top-10 ADRs:", top_adrs)

# -------------------------------
# Build contingency table for chi-square
# -------------------------------

# Combine all Twitter data
twitter_combined = Counter()
for row in df_tw["Extracted_ADRs"]:
    twitter_combined.update(row)
total_tw = sum(twitter_combined.values())
twitter_counts = [twitter_combined.get(adr, 0) for adr in top_adrs]


# Combine all FDA data
fda_combined = Counter()
for v in fda_info.values():
    fda_combined.update(load_fda(*v))
total_fda = sum(fda_combined.values())
total_tw = sum(twitter_combined.values())
# Total count of ADRs in Twitter (real count)
total_tw = sum(twitter_combined.values())
print("Total Twitter ADRs:", total_tw)
 #Step 2: # Re-scale FDA proportions to match Twitter scale ***
fda_counts = [fda_combined.get(adr, 0) * total_tw for adr in top_adrs]
#fda_counts = [fda_combined.get(adr, 0) * 1000 for adr in top_adrs]  # Multiply for scale

# Create contingency table
observed = [fda_counts, twitter_counts]

# -------------------------------
# Run chi-square test
# -------------------------------
chi2, p, dof, expected = stats.chi2_contingency(observed)

# -------------------------------
# Output results
# -------------------------------
print("\n--- Chi-Square Test Results ---")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.6f}")

# Optional: Display expected counts
expected_df = pd.DataFrame(expected, columns=top_adrs, index=["FDA", "Twitter"])
print("\nExpected Frequencies:\n", expected_df.round(2))
expected_df.to_csv("Results/ChiSquare_Expected_Frequencies.csv")

