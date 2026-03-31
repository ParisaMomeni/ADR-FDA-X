# ============================================
# origami_star_coarse_level_fda_twitter.py
# ============================================
# ONE plot — aggregate (coarse-level) FDA vs Twitter
# - Combines all brands and doses
# - Normalized to [0,1]
# - Closed rings + shared grid
# - Axes = Top-10 global ADRs across both sources
# ============================================

import warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

warnings.filterwarnings("ignore")


plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
})

EPSILON = 1e-6

# -----------------------------
# Geometry helpers
# -----------------------------
def _lty_map(r_lty):
    return {1:"-", 2:"--", 3:":", 4:"-.", 5:(0,(1,1)), 6:(0,(3,1,1,1))}.get(r_lty, "-")

def _origami_angles(n):
    # R: theta <- seq(90, 450, length=n+1)[1:n] * pi/180
    return np.linspace(90, 450, n+1)[:-1] * np.pi/180.0

def _interp_na_cartesian(j, vals_row, xx, yy, df_max, df_min, seg, CGap):
    n = len(vals_row)
    left = (j-1) % n
    while np.isnan(vals_row[left]) and left != j:
        left = (left-1) % n
    right = (j+1) % n
    while np.isnan(vals_row[right]) and right != j:
        right = (right+1) % n

    denomL = (df_max[left] - df_min[left]) or 1.0
    denomR = (df_max[right]- df_min[right]) or 1.0
    rL = CGap/(seg+CGap) + ((vals_row[left] - df_min[left]) / denomL) * (seg/(seg+CGap))
    rR = CGap/(seg+CGap) + ((vals_row[right]- df_min[right]) / denomR) * (seg/(seg+CGap))

    xxleft, yyleft   = xx[left]*rL,  yy[left]*rL
    xxright, yyright = xx[right]*rR, yy[right]*rR
    if xxleft > xxright:
        xxleft, xxright = xxright, xxleft
        yyleft, yyright = yyright, yyleft

    xj, yj = xx[j], yy[j]
    num = (yyleft * xxright - yyright * xxleft)
    den = (yj * (xxright - xxleft) - xj * (yyright - yyleft))
    if den == 0:
        return 0.0, 0.0
    xj_sc = xj * num / den
    yj_sc = (yj / xj) * xj_sc if xj != 0 else np.sign(yj) * abs(xj_sc)
    return xj_sc, yj_sc

def _scale_row_to_cartesian(vals_row, xx, yy, df_max, df_min, seg, CGap, na_itp=True):
    n = len(vals_row)
    xxs = np.zeros(n); yys = np.zeros(n)
    denom = (df_max - df_min).copy()
    denom[denom == 0] = 1.0
    base_r = CGap/(seg+CGap) + ((vals_row - df_min) / denom) * (seg/(seg+CGap))
    for j in range(n):
        if np.isnan(vals_row[j]):
            if na_itp:
                xxs[j], yys[j] = _interp_na_cartesian(j, vals_row, xx, yy, df_max, df_min, seg, CGap)
            else:
                xxs[j], yys[j] = 0.0, 0.0
        else:
            xxs[j], yys[j] = xx[j]*base_r[j], yy[j]*base_r[j]
    return xxs, yys

def _to_star_path_from_xy(xxs, yys, inner_radius, theta):
    n = len(xxs)
    star_x, star_y = [], []
    for j in range(n):
        star_x.append(xxs[j]); star_y.append(yys[j])   # tip
        t_curr = theta[j]; t_next = theta[(j + 1) % n]
        dt = (t_next - t_curr + np.pi) % (2*np.pi) - np.pi
        t_mid = t_curr + dt / 2.0
        star_x.append(inner_radius * np.cos(t_mid))
        star_y.append(inner_radius * np.sin(t_mid))
    star_x.append(star_x[0]); star_y.append(star_y[0])  # close
    return np.array(star_x), np.array(star_y)

# -----------------------------
# Plotter (rings closed & grid on top)
# -----------------------------
def origami_star_plot(
    df, seg=10, cglty=3, cglwd=1, cglcol="#c9c9c9",
    title="", na_itp=True, centerzero=False, vlabels=None, vlcex=None,
    pcol=None, plty=None, plwd=1, pfcol=None,
    ax=None,
    inner_r_override=None,
    return_handles=False,
    draw_spokes_from_center=True
):
    n = df.shape[1]
    theta = _origami_angles(n)
    xx, yy = np.cos(theta), np.sin(theta)
    CGap = 0 if centerzero else 1

    series = df.shape[0] - 2
    if pcol is None:  pcol  = ["#00a19a", "#ff2d7a", "C2", "C3"]
    if plty is None:  plty  = [1, 2, 1, 2]
    if pfcol is None: pfcol = ["#00a19a22", "#ff2d7a22", None, None]

    pcol  = (pcol  * ((series + len(pcol)-1)//len(pcol)))[:series]
    plty  = (plty  * ((series + len(plty)-1)//len(plty)))[:series]
    pfcol = (pfcol * ((series + len(pfcol)-1)//len(pfcol)))[:series]

    created = False
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111)
        created = True

    ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, pad=32, fontsize=16, fontweight="bold")

    # Dynamic rings & frame
    max_val  = float(np.nanmax(df.iloc[2:].to_numpy()))
    max_ring = float(np.ceil(max_val * 10) / 10.0)
    if max_ring < 0.1: max_ring = 0.1
    ticks = np.arange(0.1, max_ring + 1e-9, 0.1)
    def rmap(v): return (CGap/(seg+CGap) + v * (seg/(seg+CGap)))
    frame = rmap(max_ring)
    ax.set_xlim(-frame, frame); ax.set_ylim(-frame, frame)

    # Data (under grid)
    df_max = df.iloc[0, :].astype(float).values
    df_min = df.iloc[1, :].astype(float).values
    inner_r = (CGap / (seg + CGap)) if inner_r_override is None else float(inner_r_override)

    line_handles, line_labels = [], []
    for idx in range(series):
        vals = df.iloc[idx+2, :].astype(float).values
        xxs, yys = _scale_row_to_cartesian(vals, xx, yy, df_max, df_min, seg, CGap, na_itp=na_itp)
        sx, sy = _to_star_path_from_xy(xxs, yys, inner_r, theta)
        ax.fill(sx, sy, color=(pfcol[idx] if pfcol[idx] else "none"), zorder=2)
        lw = plwd if isinstance(plwd, (int, float)) else plwd[idx]
        line, = ax.plot(
            sx, sy, color=pcol[idx], linewidth=lw, linestyle=_lty_map(plty[idx]),
            marker="o", markersize=3, solid_capstyle="round", solid_joinstyle="round",
            zorder=3
        )
        if return_handles:
            line_handles.append(line); line_labels.append(df.index[idx+2])

    # GRID ON TOP — spokes and CLOSED rings
    for k in range(n):
        x0, y0 = (0.0, 0.0) if draw_spokes_from_center else (xx[k]/(seg+CGap), yy[k]/(seg+CGap))
        ax.plot([x0, xx[k]*frame], [y0, yy[k]*frame],
                linestyle=_lty_map(cglty), linewidth=cglwd, color=cglcol, zorder=5)

    for rv in ticks:
        r_geom = rmap(rv)
        # close ring explicitly to avoid tiny gap at seam
        xring = np.r_[xx, xx[0]] * r_geom
        yring = np.r_[yy, yy[0]] * r_geom
        ax.plot(xring, yring, linestyle=_lty_map(cglty), linewidth=cglwd, color=cglcol, zorder=5)
        ax.text(-0.060*frame, r_geom, f"{rv:.1f}", ha="right", va="center",
                color="#4a4a4a", fontsize=13, zorder=6)

    labels = list(df.columns) if vlabels is None else list(vlabels)
    for k in range(n):
        ax.text(xx[k]*frame*1.05, yy[k]*frame*1.05, labels[k],
                ha="center", va="center", fontsize=(16 if vlcex is None else 16*vlcex),
                zorder=8)

    if created:
        plt.tight_layout(); plt.show()

    return (ax, line_handles, line_labels) if return_handles else ax
   

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
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

# ------------------------------------------------------
# Aggregate ALL Twitter ADRs (no brand distinction)
# ------------------------------------------------------
all_adrs = [adr for row in df_tw["Extracted_ADRs"] for adr in row]
twitter_all = Counter(all_adrs)
tot_twitter = sum(twitter_all.values())
twitter_all = {k: v/tot_twitter for k,v in twitter_all.items()} if tot_twitter else {}

# ------------------------------------------------------
# Aggregate ALL FDA ADRs across brands/doses
# ------------------------------------------------------
fda_all = Counter()
for v in fda_info.values():
    fda_all.update(load_fda(*v)) # population-level ADR distribution across all semaglutide discourse
tot_fda = sum(fda_all.values())
fda_all = {k:v/tot_fda for k,v in fda_all.items()} if tot_fda else {}

# ------------------------------------------------------
# Top-10 ADRs globally
# ------------------------------------------------------
combined = Counter()
combined.update(twitter_all)
combined.update(fda_all)
top_adrs = [adr for adr, _ in combined.most_common(10)]

# ------------------------------------------------------
# Build dataframe for origami star
# ------------------------------------------------------
def combined_df_for_origami(axes, twitter_dict, fda_dict, eps=EPSILON):
    t = np.array([twitter_dict.get(a, 0.0) for a in axes], float)
    f = np.array([fda_dict.get(a, 0.0) for a in axes], float)
    for arr in (t, f):
        arr[:] = np.maximum(arr, eps)
        s = arr.sum()
        if s > 0: arr[:] = arr / s
    df = pd.DataFrame([np.ones(len(axes)), np.zeros(len(axes)), t, f], columns=axes)
    df.index = ["max", "min", "X", "FDA"]
    return df

df_coarse = combined_df_for_origami(top_adrs, twitter_all, fda_all)
#add stat:

# Extract height vectors for metrics
h_twitter = df_coarse.loc["X"].values.astype(float)
h_fda     = df_coarse.loc["FDA"].values.astype(float)

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

# Compute metrics
iou_coarse = compute_iou(h_twitter, h_fda)
asd_t_f_coarse, asd_f_t_coarse = compute_asd(h_twitter, h_fda)

# Print for record
print("\n=== Coarse-Level Metrics ===")
print(f"IoU: {iou_coarse:.4f}")
print(f"ASD(Twitter|FDA): {asd_t_f_coarse:.4f}")
print(f"ASD(FDA|Twitter): {asd_f_t_coarse:.4f}")

# ------------------------------------------------------
# ------------------------------------------------------
# Coarse-Level: FDA vs Twitter (All Semaglutide ADRs)
# ------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))  # ✅ Create single axes correctly

pcol = ["#00a19a", "#ff2d7a"]
plty = [1, 2]
plwd = [2, 2]
pfcol = ["#00a19a11", "#ff2d7a11"]  # same faint fill

# ✅ Plot onto the single axes
ax, handles, labels = origami_star_plot(
    df_coarse,
    seg=10,
    cglty=3,
    cglwd=1,
    cglcol="#cfcfcf",
    #title="Coarse-Level: FDA vs Twitter (All Semaglutide ADRs)",
    title=f"Coarse-Level: FDA vs X (All Semaglutide ADRs)\n(IoU = {iou_coarse:.3f}, ASD = {asd_t_f_coarse:.3f})",
    centerzero=False,
    vlabels=top_adrs,
    pcol=pcol,
    plty=plty,
    plwd=plwd,
    pfcol=pfcol,
    ax=ax,                  # ✅ Send correct ax object
    return_handles=True,
    draw_spokes_from_center=True
)

# ✅ Move legend to top center
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=2,
    frameon=True,
    framealpha=0.95,
    edgecolor="#DDDDDD",
    fontsize=14,
    bbox_to_anchor=(0.5, 1.02)
)

fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("Results/coarse_level_origami_grid.png", dpi=300, bbox_inches="tight")
plt.show()


