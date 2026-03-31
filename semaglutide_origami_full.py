"""
semaglutide_origami_full.py  (paper-style origami, inline + save, single file)

What it does
------------
- Loads your Twitter ADR data and FDA tables
- Picks the global Top-10 ADRs as axes
- Computes origami metrics (IoU, ASD, KL/JS/Hellinger, Jaccards, areas)
- Draws a "paper-style" origami chart (polar spider grid with labels) showing
  Twitter vs FDA filled polygons, with triangle markers on main axes
- Shows each chart inline (plt.show) and saves PNGs to Results/
- Writes Results/origami_summary.csv and Results/top10_axes.txt

Inputs (same as your previous pipeline)
---------------------------------------
- data/1adrs2.pkl                  (Twitter dataframe with Extracted_ADRs, Title, Snippet)
- data/FDAWegavy.xlsx              (column: "FDA 2.4 mg")
- data/FDAOzempic.xlsx             (columns: "FDA 0.5 mg", "FDA 1 mg")
- data/FDARybelsus.xlsx            (columns: "FDA 7 mg", "FDA 14 mg")

Dependencies
------------
pip install numpy pandas matplotlib shapely scipy openpyxl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.stats import entropy

from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable

# ----------------------
# Config
# ----------------------
EPSILON = 1e-6
RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUX_RADIUS = 0.5          # try 0.3–0.7
WEIGHTS: Optional[np.ndarray] = None  # optional length-n weights that sum to 1
TOP_N_AXES = 10           # <- exactly 10 ADRs for axes

# Data locations
TWEETS_PATH = "data/1adrs2.pkl"
FDA_INFO = {
    "Wegovy":      ("data/FDAWegavy.xlsx",   "FDA 2.4 mg"),
    "Ozempic_0.5": ("data/FDAOzempic.xlsx",  "FDA 0.5 mg"),
    "Ozempic_1":   ("data/FDAOzempic.xlsx",  "FDA 1 mg"),
    "Rybelsus_7":  ("data/FDARybelsus.xlsx", "FDA 7 mg"),
    "Rybelsus_14": ("data/FDARybelsus.xlsx", "FDA 14 mg"),
}
LABELS_TO_RUN: List[Tuple[str, str]] = [
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

# ============================================================
#               ORIGAMI CORE (metrics + geometry)
# ============================================================

def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    ps = p / p.sum() if p.sum() else p
    qs = q / q.sum() if q.sum() else q
    return float((1/np.sqrt(2)) * np.linalg.norm(np.sqrt(ps) - np.sqrt(qs)))

def _weighted_jaccard(p: np.ndarray, q: np.ndarray) -> float:
    num = np.minimum(p, q).sum()
    den = np.maximum(p, q).sum()
    return 0.0 if den == 0 else float(num / den)

def _build_cartesian_origami_points(r_main: np.ndarray,
                                    aux_radius: float,
                                    weights: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
    """
    Build a Cartesian origami polygon (closed) with 2n vertices by interleaving
    main-axis vertices (scaled radii) and auxiliary mid-axis vertices (fixed radius).
    """
    r = r_main.astype(float).copy()
    n = r.size
    if weights is not None:
        w = np.asarray(weights, float)
        w = w / w.sum() if w.sum() else w
        r *= (w / (1.0/n))
    r = np.maximum(r, EPSILON)
    theta_main = np.linspace(0, 2*np.pi, n, endpoint=False)
    dtheta = 2*np.pi / n
    out: List[Tuple[float, float]] = []
    for i in range(n):
        out.append((float(r[i]*np.cos(theta_main[i])), float(r[i]*np.sin(theta_main[i]))))
        aux_ang = theta_main[i] + dtheta/2.0
        out.append((float(aux_radius*np.cos(aux_ang)), float(aux_radius*np.sin(aux_ang))))
    out.append(out[0])
    return out

def compare_origami_metrics(r_t: np.ndarray,
                            r_f: np.ndarray,
                            aux_radius: float = 0.5,
                            weights: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], List[Tuple[float,float]], List[Tuple[float,float]]]:
    """
    Compute origami polygons (Cartesian) + metrics.
    Returns (metrics_dict, twitter_poly_xy, fda_poly_xy)
    """
    # ε-clip + renormalize distributions for distance metrics
    p = np.maximum(r_t.astype(float), EPSILON); p /= p.sum()
    q = np.maximum(r_f.astype(float), EPSILON); q /= q.sum()

    poly_t = _build_cartesian_origami_points(p, aux_radius=aux_radius, weights=weights)
    poly_f = _build_cartesian_origami_points(q, aux_radius=aux_radius, weights=weights)

    A = ShapelyPolygon(poly_t); B = ShapelyPolygon(poly_f)
    if (not A.is_valid) or (not B.is_valid) or A.area <= 0 or B.area <= 0:
        geom = dict(intersection=0.0, union=0.0, iou=0.0, asd=0.0, area_ratio=0.0)
    else:
        inter = A.intersection(B).area
        union = A.union(B).area
        geom = dict(
            intersection=float(inter),
            union=float(union),
            iou=float(inter/union if union>0 else 0.0),
            asd=float(inter/B.area),
            area_ratio=float(min(A.area, B.area)/max(A.area, B.area))
        )

    # Probability distances
    kl = float(entropy(p, q))
    kl_bits = float(entropy(p, q, base=2))
    kl_rev = float(entropy(q, p))
    m = 0.5*(p+q)
    js = float(0.5*entropy(p, m) + 0.5*entropy(q, m))
    h = _hellinger(p, q)
    j_support = float(((p>0)&(q>0)).sum() / (((p>0)|(q>0)).sum() or 1))
    j_weighted = _weighted_jaccard(p, q)

    out = dict(**geom,
               KL=kl, KL_bits=kl_bits, KL_reverse=kl_rev, JS=js, Hellinger=h,
               Jaccard_support=j_support, Jaccard_weighted=j_weighted,
               A_twitter=float(A.area), A_fda=float(B.area),
               aux_radius=float(aux_radius), weighted=bool(weights is not None))
    return out, poly_t, poly_f

# ============================================================
#               PAPER-STYLE POLAR PLOTTING
# ============================================================

def _interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(a.size + b.size, dtype=float)
    out[0::2] = a; out[1::2] = b
    return out

def _build_polar_origami_sequence(r_main: np.ndarray, aux_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    n = r_main.size
    theta_main = np.linspace(0, 2*np.pi, n, endpoint=False)
    dtheta = 2*np.pi / n
    theta_aux = theta_main + dtheta/2.0
    r_aux = np.full(n, float(aux_radius))
    th_seq = _interleave(theta_main, theta_aux)
    r_seq = _interleave(r_main, r_aux)
    th_seq = np.append(th_seq, th_seq[0])
    r_seq = np.append(r_seq, r_seq[0])
    return th_seq, r_seq

def plot_origami_paper_two_series(labels: List[str],
                                  r_t: np.ndarray,
                                  r_f: np.ndarray,
                                  aux_radius: float,
                                  title: str):
    """
    Paper-like origami plot (polar spider), overlaying Twitter and FDA.
    """
    n = len(labels)
    assert r_t.size == n and r_f.size == n

    fig = plt.figure(figsize=(6.5, 7.2))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # light dashed radial grid at 0.25 steps up to 1.0
    ax.set_ylim(0, 1.05)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], angle=90)
    for gridline in ax.yaxis.get_gridlines():
        gridline.set_linestyle((0, (2, 3)))
        gridline.set_linewidth(0.8)
        gridline.set_alpha(0.5)

    theta_main = np.linspace(0, 2*np.pi, n, endpoint=False)
    ax.set_thetagrids(np.degrees(theta_main), labels)

    # Build sequences and plot
    th_t, rseq_t = _build_polar_origami_sequence(r_t, aux_radius=aux_radius)
    th_f, rseq_f = _build_polar_origami_sequence(r_f, aux_radius=aux_radius)

    ax.plot(th_t, rseq_t, linewidth=2, label="Twitter")
    ax.fill(th_t, rseq_t, alpha=0.20)
    ax.plot(theta_main, r_t, marker="^", linestyle="None", markersize=6)

    ax.plot(th_f, rseq_f, linewidth=1, label="FDA")
    ax.fill(th_f, rseq_f, alpha=0.20)
    ax.plot(theta_main, r_f, marker="^", linestyle="None", markersize=6)

    ax.legend(loc="upper right")
    if title:
        ax.set_title(title, va='bottom')
    fig.tight_layout()
    return fig

# ============================================================
#               DATA LOADING & AXIS PREP
# ============================================================

def _extract_brand(text: str) -> str:
    t = (text or "").lower()
    if "wegovy" in t:   return "Wegovy"
    if "ozempic" in t:  return "Ozempic"
    if "rybelsus" in t: return "Rybelsus"
    return "Unknown"

def load_twitter_data(pickle_path: str) -> pd.DataFrame:
    df = pd.read_pickle(pickle_path)
    df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else [])
    df["Brand"] = (df["Title"].fillna('') + ' ' + df["Snippet"].fillna('')).apply(_extract_brand)
    return df

def load_fda_shares(path: str, col: str) -> Dict[str, float]:
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: "ADR", col: "Prevalence"})
    df = df[["ADR", "Prevalence"]].dropna()
    df["ADR"] = df["ADR"].str.lower()
    vals = df["Prevalence"].astype(str).str.replace('%', '', regex=False).astype(float)
    total = vals.sum()
    shares = (vals / total) if total else pd.Series(np.zeros(len(vals)))
    return dict(zip(df["ADR"], shares))

def twitter_counts_and_shares(df_tweets: pd.DataFrame) -> Tuple[Dict[str, Counter], Dict[str, Dict[str, float]]]:
    counts_raw: Dict[str, Counter] = {}
    shares: Dict[str, Dict[str, float]] = {}
    for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
        adrs = [adr for row in df_tweets[df_tweets["Brand"] == brand]["Extracted_ADRs"] for adr in row]
        c = Counter(adrs)
        counts_raw[brand] = c
        total = sum(c.values())
        shares[brand] = {k: v / total for k, v in c.items()} if total else {}
    return counts_raw, shares

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

# ============================================================
#               ORCHESTRATION
# ============================================================

def run_pipeline(top_n_axes: int = TOP_N_AXES,
                 aux_radius: float = AUX_RADIUS,
                 weights: Optional[np.ndarray] = WEIGHTS) -> pd.DataFrame:
    print("Loading Twitter data from:", TWEETS_PATH)
    df = load_twitter_data(TWEETS_PATH)

    print("Computing Twitter counts and shares...")
    _, twitter_shares = twitter_counts_and_shares(df)

    print(f"Selecting Top-{top_n_axes} ADR axes...")
    top_adrs = compute_top_adrs(twitter_shares, n_axes=top_n_axes)
    print("Axes:", top_adrs)
    (RESULTS_DIR / "top10_axes.txt").write_text("\n".join(top_adrs))

    results = []
    for brand, fda_label in LABELS_TO_RUN:
        fda_key = BRAND_TO_FDA_KEY[(brand, fda_label)]
        print(f"\n=== {brand} vs {fda_label} ===")
        fda_shares = load_fda_shares(*FDA_INFO[fda_key])

        r_t = shares_to_vector(twitter_shares.get(brand, {}), top_adrs)
        r_f = shares_to_vector(fda_shares, top_adrs)

        # Metrics (Cartesian origami)
        metrics, _, _ = compare_origami_metrics(r_t, r_f, aux_radius=aux_radius, weights=weights)

        # Paper-style plot (polar) overlaying the two series
        fig = plot_origami_paper_two_series(
            labels=top_adrs,
            r_t=r_t,
            r_f=r_f,
            aux_radius=aux_radius,
            title=f"{brand} vs {fda_label}"
        )

        safe_brand = brand.lower().replace(" ", "_")
        safe_label = fda_label.lower().replace(" ", "_").replace(".", "").replace("/", "-")
        out_png = RESULTS_DIR / f"{safe_brand}_{safe_label}_origami_paper.png"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print("Saved figure:", out_png)

        metrics["Brand"] = brand
        metrics["FDA"] = fda_label
        metrics["axes"] = "|".join(top_adrs)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "origami_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print("\nWrote metrics CSV:", out_csv)
    return results_df

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
