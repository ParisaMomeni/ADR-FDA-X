# ============================================
# metrics_from_twitter_fda_dynamic.py
# ============================================
#to do: compute metrics between Twitter ADR distributions and FDA ADR distributions without considering brands.
import numpy as np
import pandas as pd
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import ast

# -----------------------------
# Load Twitter DF
# -----------------------------
df_tw = pd.read_pickle("Results/1adrs2_with_extractions.pkl")
def adr_str_to_list(adr_str):
    if isinstance(adr_str, str):
        try:
            return ast.literal_eval(adr_str)
        except Exception:
            return []
    elif isinstance(adr_str, list):
        return adr_str
    else:
        return []
df_tw["Extracted_ADRs"] = df_tw["Extracted_ADRs"].apply(adr_str_to_list)    
df_tw["Extracted_ADRs"] = df_tw["Extracted_ADRs"].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else [])
df_tw["Brand"] = df_tw["Primary_Brand"].fillna('')

# -----------------------------
# FDA loader & brand aggregation
# -----------------------------
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
    dfx["Prevalence"] = dfx["Prevalence"].astype(str).str.replace('%','', regex=False).astype(float)
    tot = dfx["Prevalence"].sum()
    return dict(zip(dfx["ADR"], dfx["Prevalence"]/tot)) if tot else {}

brand_level_fda = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    keys = [k for k in fda_info if k.startswith(brand)]
    acc = Counter()
    for k in keys:
        acc.update(load_fda(*fda_info[k]))
    tot = sum(acc.values())
    brand_level_fda[brand] = {k:v/tot for k,v in acc.items()} if tot else {}

# -----------------------------
# Twitter normalized ADR shares
# -----------------------------
twitter_adr_counts = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
    adrs = [adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"] for adr in row]
    cnt = Counter(adrs)
    tot = sum(cnt.values())
    twitter_adr_counts[brand] = {k: v/tot for k, v in cnt.items()} if tot else {}

# -----------------------------
# Global Top-10 ADR axes
# -----------------------------
all_counters = list(twitter_adr_counts.values()) + [load_fda(*v) for v in fda_info.values()]
combined = Counter()
for c in all_counters:
    combined.update(c)
top_adrs = [adr for adr, _ in combined.most_common(10)]

# ======================================================
# Metric Computation (IoU, ASD, Area Ratio, KL Divergence)
# ======================================================

def get_heights_for_brand(brand, top_adrs, twitter_adr_counts, brand_level_fda, eps=1e-6):
    """Extract normalized h_twitter and h_fda vectors dynamically."""
    h_twitter = np.array([twitter_adr_counts.get(brand, {}).get(a, 0.0) for a in top_adrs], dtype=float)
    h_fda     = np.array([brand_level_fda.get(brand, {}).get(a, 0.0) for a in top_adrs], dtype=float)

    h_twitter = np.maximum(h_twitter, eps)
    h_fda     = np.maximum(h_fda, eps)

    #if renorm:
     #   h_twitter /= np.sum(h_twitter)
     #   h_fda     /= np.sum(h_fda)

    return h_twitter, h_fda


def compute_iou(h_twitter, h_fda):
    """Intersection-over-Union between two normalized ADR vectors."""
    num = np.sum(np.minimum(h_twitter, h_fda))
    den = np.sum(np.maximum(h_twitter, h_fda))
    return num / den if den > 0 else np.nan

def compute_asd(h_twitter, h_fda):
    """Asymmetric Similarity Degree (Jeong et al. 2019)."""
    num = np.sum(np.minimum(h_twitter, h_fda))
    den_t_f = np.sum(h_fda)
    den_f_t = np.sum(h_twitter)
    asd_t_f = num / den_t_f if den_t_f > 0 else np.nan  # ASD(Twitter | FDA)
    asd_f_t = num / den_f_t if den_f_t > 0 else np.nan  # ASD(FDA | Twitter)
    return asd_t_f, asd_f_t


def compute_area_ratio(h_twitter, h_fda):
    """Ratio of Twitter area to FDA area (sum of heights as proxy)."""
    s_t = np.sum(h_twitter)
    s_f = np.sum(h_fda)
    return min(s_t, s_f) / max(s_t, s_f) if max(s_t, s_f) > 0 else np.nan


def compute_kl_divergence(h_twitter, h_fda):
    """Kullback–Leibler Divergence (symmetric version)."""
    kl_tf = np.sum(h_twitter * np.log(h_twitter / h_fda))
    kl_ft = np.sum(h_fda * np.log(h_fda / h_twitter))
    # Compute normalized similarities (invert + normalize)
    skl_tf = 1 / (1 + kl_tf)
    skl_ft = 1 / (1 + kl_ft)

    return kl_tf, kl_ft, skl_tf, skl_ft


# ======================================================
# Run metrics per brand
# ======================================================
brands = ["Ozempic", "Wegovy", "Rybelsus"]
results = []

for brand in brands:
    h_twitter, h_fda = get_heights_for_brand(brand, top_adrs, twitter_adr_counts, brand_level_fda)
    iou = compute_iou(h_twitter, h_fda)
    asd_t_f, asd_f_t = compute_asd(h_twitter, h_fda)
    ar  = compute_area_ratio(h_twitter, h_fda)
    kl_tf, kl_ft, skl_tf, skl_ft = compute_kl_divergence(h_twitter, h_fda)


    results.append({
        "Brand": brand,
        "IoU": iou,
        "ASD_Twitter|FDA": asd_t_f,
        "ASD_FDA|Twitter": asd_f_t,
        "AreaRatio": ar,
        "KL_Divergence_Twitter|FDA": kl_tf,
        "KL_Divergence_FDA|Twitter": kl_ft,
        "S_KL_Twitter|FDA": skl_tf,
        "S_KL_FDA|Twitter": skl_ft
    })

# -----------------------------
# Save + display results
# -----------------------------
df_metrics = pd.DataFrame(results)
df_metrics = df_metrics.round(4)
print(df_metrics)

df_metrics.to_csv("Results/twitter_fda_metrics.csv", index=False)
print("\n✅ Metrics saved to Results/twitter_fda_metrics.csv")

# ======================================================
# Visualization Section
# ======================================================

sns.set_theme(style="whitegrid", font_scale=1.2)

# --- 1. IoU Bar Chart ---
plt.figure(figsize=(7, 5))
sns.barplot(x="Brand", y="IoU", data=df_metrics, color="#00a19a", alpha=0.8)
plt.title("Intersection-over-Union (IoU): Twitter vs FDA", fontsize=14)
plt.ylabel("IoU Value")
for i, v in enumerate(df_metrics["IoU"]):
    plt.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("Results/iou_bar_chart.png", dpi=300)
plt.show()

# --- 2. Multi-Metric Radar Chart ---
metrics = [
    "IoU", "ASD_Twitter|FDA", "ASD_FDA|Twitter",
    "AreaRatio", "KL_Divergence_Twitter|FDA", "KL_Divergence_FDA|Twitter"
]
df_norm = df_metrics.copy()
for col in metrics:
    if "KL" in col:
        df_norm[col] = 1 / (1 + df_metrics[col])
    else:
        df_norm[col] = df_metrics[col] / df_metrics[col].max()

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]

plt.figure(figsize=(7, 7))
for _, row in df_norm.iterrows():
    values = row[metrics].tolist() + [row[metrics[0]]]
    plt.polar(angles, values, label=row["Brand"], linewidth=2, alpha=0.85)
plt.xticks(angles[:-1], metrics, fontsize=11)
plt.title("Alignment Between Twitter and FDA Across Metrics", fontsize=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig("Results/radar_chart.png", dpi=300)
plt.show()

# --- 3. Heatmap Summary ---
plt.figure(figsize=(8, 3))
sns.heatmap(df_metrics.set_index("Brand")[metrics], annot=True, cmap="RdYlBu_r",
            fmt=".2f", cbar_kws={"label": "Metric Value"})
plt.title("Metric Comparison Between Twitter and FDA", fontsize=14)
plt.tight_layout()
plt.savefig("Results/metrics_heatmap.png", dpi=300)
plt.show()


# Normalize metrics (invert KL so larger means better alignment)
df_norm = df_metrics.copy()
for col in df_metrics.columns[1:]:
    if "KL" in col:
        df_norm[col] = 1 / (1 + df_metrics[col])
    else:
        df_norm[col] = df_metrics[col] / df_metrics[col].max()

# --- Prepare nodes and links for the chord plot ---
brands = df_norm["Brand"].tolist()
metrics = df_norm.columns[1:].tolist()

# Define all nodes
nodes = brands + metrics
node_colors = (
    ["#2E8B57", "#4682B4", "#D2691E"]  # brand colors
    + ["#C0C0C0"] * len(metrics)       # metric colors
)

# Define links
source_indices, target_indices, values = [], [], []
for i, brand in enumerate(brands):
    for j, metric in enumerate(metrics):
        source_indices.append(i)
        target_indices.append(len(brands) + j)
        values.append(df_norm.loc[i, metric])

# --- Build chord-like Sankey diagram ---
fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors,
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=[
                    f"rgba(0,100,200,{v})" for v in np.interp(values, (min(values), max(values)), (0.3, 0.9))
                ],
            ),
        )
    ]
)

fig.update_layout(
    title_text="Arc (Chord) Visualization of Metric Alignment — Twitter vs FDA",
    font_size=12,
    font_family="Times New Roman",
    height=600,
    width=950,
)

fig.show()