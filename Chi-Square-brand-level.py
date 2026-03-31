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
#new:
non_empty_adr_count = df_tw["Extracted_ADRs"].apply(lambda x: len(x) > 0).sum()
print("Rows with at least one ADR:", non_empty_adr_count)

print(df_tw.head(100))

print("Number of rows after filtering:", len(df_tw))


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
    tot = dfx["Prevalence"].sum()
    return dict(zip(dfx["ADR"], dfx["Prevalence"] / tot)) if tot else {}

# -------------------------------
# Aggregate FDA values to brand level
# -------------------------------
brand_level_fda = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    keys = [k for k in fda_info if k.startswith(brand)]
    acc = Counter()
    for k in keys:
        acc.update(load_fda(*fda_info[k]))
    tot = sum(acc.values())
    brand_level_fda[brand] = {k: v / tot for k, v in acc.items()} if tot else {}

# -------------------------------
# Aggregate Twitter ADRs per brand
# -------------------------------
twitter_adr_counts = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    adrs = [
        adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"]
        for adr in row
    ]
    cnt = Counter(adrs)
    tot = sum(cnt.values())
    twitter_adr_counts[brand] = {k: v / tot for k, v in cnt.items()} if tot else {}

# -------------------------------
# Identify top 10 ADRs across all sources
# -------------------------------
all_counters = list(twitter_adr_counts.values()) + list(brand_level_fda.values())
combined = Counter()
for c in all_counters:
    combined.update(c)
top_adrs = [a for a, _ in combined.most_common(10)]
print("Top-10 ADRs:", top_adrs)

# -------------------------------
# Run chi-square for each brand
# -------------------------------

print("\n--- Brand-Level Chi-Square Test Results ---")
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    print(f"\nBrand: {brand}")

    twitter_dist = twitter_adr_counts.get(brand, {})
    fda_dist = brand_level_fda.get(brand, {})

    twitter_total = sum(Counter([
        adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"]
        for adr in row
    ]).values())

    # Raw counts (not normalized)
    twitter_counts = [twitter_dist.get(adr, 0) * twitter_total for adr in top_adrs]
    fda_counts = [fda_dist.get(adr, 0) * 100000 for adr in top_adrs]

    # Filter out any columns where either row is zero
    filtered_data = [
        (fda, tw) for fda, tw in zip(fda_counts, twitter_counts) if fda > 0 and tw > 0
    ]

    if len(filtered_data) < 2:
        print("  Not enough data to run chi-square test.")
        continue

    fda_filtered, twitter_filtered = zip(*filtered_data)
    observed = [fda_filtered, twitter_filtered]

    try:
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        print(f"  Chi-square statistic: {chi2:.2f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  P-value: {p:.6f}")
    except ValueError as e:
        print(f"  Error in chi-square test: {e}")
