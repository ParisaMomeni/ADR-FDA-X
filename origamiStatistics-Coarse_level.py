# ======================================================
# Coarse-Level (All Semaglutide) Metrics
# ======================================================
# This script computes similarity/divergence metrics
# between *aggregated* Twitter ADR distributions
# and *aggregated* FDA ADR distributions — without
# distinguishing between brands, doses, or users.
# ======================================================

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import ast
# -----------------------------
# Load Twitter Data
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
df_tw["Extracted_ADRs"] = df_tw["Extracted_ADRs"].apply(
    lambda x: [i.lower() for i in x] if isinstance(x, list) else []
)

# -----------------------------
# FDA Loader
# -----------------------------
fda_info = {
    "Wegovy":      ("data/FDAWegavy.xlsx", "FDA 2.4 mg"),
    "Ozempic_0.5": ("data/FDAOzempic.xlsx", "FDA 0.5 mg"),
    "Ozempic_1":   ("data/FDAOzempic.xlsx", "FDA 1 mg"),
    "Rybelsus_7":  ("data/FDARybelsus.xlsx", "FDA 7 mg"),
    "Rybelsus_14": ("data/FDARybelsus.xlsx", "FDA 14 mg"),
}

def load_fda(path, col):
    """Load and normalize ADR prevalence data from FDA spreadsheets."""
    dfx = pd.read_excel(path)
    dfx = dfx.rename(columns={dfx.columns[0]: "ADR", col: "Prevalence"})
    dfx = dfx[["ADR", "Prevalence"]].dropna()
    dfx["ADR"] = dfx["ADR"].str.lower()
    dfx["Prevalence"] = (
        dfx["Prevalence"].astype(str).str.replace("%", "", regex=False).astype(float)
    )
    tot = dfx["Prevalence"].sum()
    return dict(zip(dfx["ADR"], dfx["Prevalence"] / tot)) if tot else {}

# -----------------------------
# Aggregate Twitter ADRs (no brands)
# -----------------------------
twitter_counts = Counter([adr for row in df_tw["Extracted_ADRs"] for adr in row])

# -----------------------------
# Aggregate FDA ADRs (across all brands/doses)
# -----------------------------
fda_all = Counter()
for v in fda_info.values():
    fda_all.update(load_fda(*v))

# -----------------------------
# Unified ADR Axis
# -----------------------------
all_adrs_union = sorted(set(list(twitter_counts.keys()) + list(fda_all.keys())))

EPS = 1e-6
# Raw vectors
h_twitter_all = np.array([twitter_counts.get(a, 0.0) for a in all_adrs_union], dtype=float)
h_fda_all     = np.array([fda_all.get(a, 0.0) for a in all_adrs_union], dtype=float)

# Normalized for shape-based metrics
h_twitter_norm = h_twitter_all / np.sum(h_twitter_all)
h_fda_norm     = h_fda_all / np.sum(h_fda_all)
h_twitter_norm = np.maximum(h_twitter_norm, EPS)
h_fda_norm     = np.maximum(h_fda_norm, EPS)

# ======================================================
# Metric Computation
# ======================================================
def compute_iou(h_twitter, h_fda):
    """Intersection-over-Union (IoU) between two normalized ADR vectors."""
    num = np.sum(np.minimum(h_twitter, h_fda))
    den = np.sum(np.maximum(h_twitter, h_fda))
    return num / den if den > 0 else np.nan

def compute_asd(h_twitter, h_fda):
    """Asymmetric Similarity Degree (Jeong et al., 2019)."""
    num = np.sum(np.minimum(h_twitter, h_fda))
    den_t_f = np.sum(h_fda)
    den_f_t = np.sum(h_twitter)
    asd_t_f = num / den_t_f if den_t_f > 0 else np.nan  # ASD(Twitter | FDA)
    asd_f_t = num / den_f_t if den_f_t > 0 else np.nan  # ASD(FDA | Twitter)
    return asd_t_f, asd_f_t

def compute_area_ratio(h_twitter, h_fda):
    """Ratio of total ADR area (proxy for reporting volume)."""
    s_t, s_f = np.sum(h_twitter), np.sum(h_fda)
    return min(s_t, s_f) / max(s_t, s_f) if max(s_t, s_f) > 0 else np.nan

def compute_kl_divergence(h_twitter, h_fda):
    """Kullback–Leibler Divergence and Similarity (S_KL)."""
    # Normalize inside the function to ensure valid probability distributions
    p = h_twitter / np.sum(h_twitter)
    q = h_fda / np.sum(h_fda)
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    kl_tf = np.sum(p * np.log(p / q))
    kl_ft = np.sum(q * np.log(q / p))
    skl_tf = 1 / (1 + kl_tf)
    skl_ft = 1 / (1 + kl_ft)
    return kl_tf, kl_ft, skl_tf, skl_ft

# ======================================================
# Compute Metrics for Coarse Level
# ======================================================
iou_all = compute_iou(h_twitter_norm, h_fda_norm)
asd_t_f_all, asd_f_t_all = compute_asd(h_twitter_norm, h_fda_norm)
kl_tf_all, kl_ft_all, skl_tf_all, skl_ft_all = compute_kl_divergence(h_twitter_norm, h_fda_norm)
ar_all = compute_area_ratio(h_twitter_all, h_fda_all)

# ======================================================
# Results Table
# ======================================================
results = [{
    "Level": "All_Semaglutide (Coarse)",
    "IoU": iou_all,
    "ASD_Twitter|FDA": asd_t_f_all,
    "ASD_FDA|Twitter": asd_f_t_all,
    "AreaRatio": ar_all,
    "KL_Divergence_Twitter|FDA": kl_tf_all,
    "KL_Divergence_FDA|Twitter": kl_ft_all,
    "S_KL_Twitter|FDA": skl_tf_all,
    "S_KL_FDA|Twitter": skl_ft_all
}]

df_metrics = pd.DataFrame(results).round(4)
print("\n===== Coarse-Level (All Semaglutide) Results =====")
print(df_metrics.to_string(index=False))
df_metrics.to_csv("Results/coarse_level_twitter_fda_metrics.csv", index=False)
print("\n✅ Saved to Results/coarse_level_twitter_fda_metrics.csv")
