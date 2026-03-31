"""
semaglutide_origami_pipeline.py  (inline + save, single file)

Requirements:
- numpy, pandas, matplotlib, shapely, scipy
- origami_utils.py available on PYTHONPATH or in the same folder.

Outputs:
- Results/*_origami.png
- Results/origami_summary.csv
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from origami_utils import compare_origami  # <-- keep origami_utils.py alongside this file

# ----------------------
# Config
# ----------------------
EPSILON = 1e-6
RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUX_RADIUS = 0.5       # try 0.3–0.7
WEIGHTS = None         # or a length-n array summing to 1 for weighted origami
TOP_N_AXES = 10

# Data
TWEETS_PATH = "data/1adrs2.pkl"
FDA_INFO = {
    "Wegovy":      ("data/FDAWegavy.xlsx",   "FDA 2.4 mg"),
    "Ozempic_0.5": ("data/FDAOzempic.xlsx",  "FDA 0.5 mg"),
    "Ozempic_1":   ("data/FDAOzempic.xlsx",  "FDA 1 mg"),
    "Rybelsus_7":  ("data/FDARybelsus.xlsx", "FDA 7 mg"),
    "Rybelsus_14": ("data/FDARybelsus.xlsx", "FDA 14 mg"),
}
LABELS: List[Tuple[str, str]] = [
    ("Wegovy", "FDA 2.4 mg"),
    ("Ozempic", "FDA 0.5 mg"),
    ("Ozempic", "FDA 1 mg"),
    ("Rybelsus", "FDA 7 mg"),
    ("Rybelsus", "FDA 14 mg"),
]
BRAND_TO_FDA_KEY = {
    ("Wegovy", "FDA 2.4 mg"): "Wegovy",
    ("Ozempic", "FDA 0.5 mg"): "Ozempic_0.5",
    ("Ozempic", "FDA 1 mg"): "Ozempic_1",
    ("Rybelsus", "FDA 7 mg"): "Rybelsus_7",
    ("Rybelsus", "FDA 14 mg"): "Rybelsus_14",
}

# ----------------------
# Inline + Save plotting helper (no extra module needed)
# ----------------------
def show_and_save_origami(r_t: np.ndarray,
                          r_f: np.ndarray,
                          title: str,
                          out_path: Path,
                          aux_radius: float = 0.5,
                          weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Build origami polygons, compute metrics, SHOW inline, and SAVE PNG.
    """
    metrics, poly_t, poly_f = compare_origami(r_t, r_f, aux_radius=aux_radius, weights=weights)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    x_t, y_t = zip(*poly_t)
    x_f, y_f = zip(*poly_f)
    ax.plot(x_t, y_t, linewidth=2, label="Twitter")
    ax.plot(x_f, y_f, linewidth=1, label="FDA")
    ax.fill(x_t, y_t, alpha=0.2)
    ax.fill(x_f, y_f, alpha=0.2)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return metrics

# ----------------------
# Data loading & prep
# ----------------------
def extract_brand(text: str) -> str:
    t = (text or "").lower()
    if "wegovy"   in t: return "Wegovy"
    if "ozempic"  in t: return "Ozempic"
    if "rybelsus" in t: return "Rybelsus"
    return "Unknown"

def load_twitter_data(pickle_path: str) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(
        lambda x: [i.lower() for i in x] if isinstance(x, list) else []
    )
    df["Brand"] = (df["Title"].fillna("") + " " + df["Snippet"].fillna("")).apply(extract_brand)
    return df

def load_fda_shares(path: str, col: str) -> Dict[str, float]:
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: "ADR", col: "Prevalence"})
    df = df[["ADR", "Prevalence"]].dropna()
    df["ADR"] = df["ADR"].str.lower()
    vals = df["Prevalence"].astype(str).str.replace("%", "", regex=False).astype(float)
    total = vals.sum()
    shares = (vals / total) if total else pd.Series(np.zeros(len(vals)))
    return dict(zip(df["ADR"], shares))

def twitter_counts_and_shares(df_tweets: pd.DataFrame) -> Tuple[Dict[str, Counter], Dict[str, Dict[str, float]]]:
    twitter_counts_raw: Dict[str, Counter] = {}
    twitter_shares: Dict[str, Dict[str, float]] = {}
    for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
        adrs = [adr for row in df_tweets[df_tweets["Brand"] == brand]["Extracted_ADRs"] for adr in row]
        count = Counter(adrs)
        twitter_counts_raw[brand] = count
        total = sum(count.values())
        twitter_shares[brand] = {k: v / total for k, v in count.items()} if total else {}
    return twitter_counts_raw, twitter_shares

def compute_top_adrs(twitter_shares: Dict[str, Dict[str, float]], n_axes: int) -> List[str]:
    all_counters: List[Counter] = list(twitter_shares.values()) + [Counter(load_fda_shares(*v)) for v in FDA_INFO.values()]
    combined = Counter()
    for c in all_counters:
        combined.update(c)
    return [adr for adr, _ in combined.most_common(n_axes)]

def shares_to_vector(shares: Dict[str, float], axes: List[str]) -> np.ndarray:
    v = np.array([shares.get(adr, 0.0) for adr in axes], dtype=float)
    v = np.maximum(v, EPSILON)
    v /= v.sum()
    return v

# ----------------------
# Orchestrator
# ----------------------
def run_pipeline(top_n_axes: int = TOP_N_AXES,
                 aux_radius: float = AUX_RADIUS,
                 weights: Optional[np.ndarray] = WEIGHTS) -> pd.DataFrame:
    print("Loading Twitter data from:", TWEETS_PATH)
    df = load_twitter_data(TWEETS_PATH)

    print("Computing Twitter counts and shares...")
    twitter_counts_raw, twitter_shares = twitter_counts_and_shares(df)

    print(f"Selecting top {top_n_axes} ADR axes...")
    top_adrs = compute_top_adrs(twitter_shares, n_axes=top_n_axes)
    print("Axes:", top_adrs)

    results = []
    for brand, fda_label in LABELS:
        fda_key = BRAND_TO_FDA_KEY[(brand, fda_label)]
        print(f"\n=== {brand} vs {fda_label} ===")
        fda_shares = load_fda_shares(*FDA_INFO[fda_key])

        r_t = shares_to_vector(twitter_shares.get(brand, {}), top_adrs)
        r_f = shares_to_vector(fda_shares, top_adrs)

        safe_brand = brand.lower().replace(" ", "_")
        safe_label = fda_label.lower().replace(" ", "_").replace(".", "").replace("/", "-")
        out_png = RESULTS_DIR / f"{safe_brand}_{safe_label}_origami.png"
        title = f"{brand} vs {fda_label}"

        metrics = show_and_save_origami(
            r_t, r_f, title=title, out_path=out_png,
            aux_radius=aux_radius, weights=weights
        )
        metrics["Brand"] = brand
        metrics["FDA"] = fda_label
        metrics["axes"] = "|".join(top_adrs)
        results.append(metrics)
        print("Saved figure:", out_png)

    results_df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "origami_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print("\nWrote metrics CSV:", out_csv)
    return results_df

# ----------------------
# main
# ----------------------
def main():
    run_pipeline()

if __name__ == "__main__":
    main()
