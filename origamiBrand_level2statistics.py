# origami_star_2x2_from_df_fda_norm_dynamic_frame_one_legend_closedrings.py
# 3 subplots (Ozempic, Wegovy, Rybelsus) — FDA vs Twitter
# - Normalized to [0,1]
# - Rings at 0.1 steps up to the brand max (rounded up)
# - Axes/frame extend ONLY to that outer ring
# - ONE shared legend
# - Grid (spokes+rings) drawn ON TOP and FROM CENTER
# - RINGS ARE EXPLICITLY CLOSED (no tiny gaps)

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
    for k in keys: acc.update(load_fda(*fda_info[k]))
    tot = sum(acc.values())
    # aggregate across doses and renormalize:
    brand_level_fda[brand] = {k:v/tot for k,v in acc.items()} if tot else {}

# -----------------------------
# Twitter normalized ADR shares
# -----------------------------
twitter_adr_counts = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
    adrs = [adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"] for adr in row]
    cnt = Counter(adrs); tot = sum(cnt.values())
    twitter_adr_counts[brand] = {k: v/tot for k, v in cnt.items()} if tot else {}

# -----------------------------
# Global Top-10 ADR axes
# -----------------------------
all_counters = list(twitter_adr_counts.values()) + [load_fda(*v) for v in fda_info.values()]
combined = Counter()
for c in all_counters: combined.update(c)
top_adrs = [adr for adr, _ in combined.most_common(10)]

# Step 2: Aggregate total counts
# --------------------------------------
# Total counts from Twitter (all brands)
twitter_agg = Counter()
for brand_data in twitter_adr_counts.values():
    twitter_agg.update({k: v for k, v in brand_data.items() if k in top_adrs})

# Total counts from FDA (all doses)
fda_agg = Counter()
for v in fda_info.values():
    fda_data = load_fda(*v)
    fda_agg.update({k: v for k, v in fda_data.items() if k in top_adrs})

# --------------------------------------
# Step 3: Normalize for bar chart
# --------------------------------------
total_twitter = sum(twitter_agg.values())
total_fda = sum(fda_agg.values())

twitter_values = [twitter_agg.get(adr, 0) / total_twitter if total_twitter else 0 for adr in top_adrs]
fda_values = [fda_agg.get(adr, 0) / total_fda if total_fda else 0 for adr in top_adrs]

# --------------------------------------
# Step 4: Bar Chart
# --------------------------------------
x = np.arange(len(top_adrs))
width = 0.45

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, twitter_values, width, label='X (Twitter)', color='#00a19a')
bars2 = ax.bar(x + width/2, fda_values, width, label='FDA', color='#ff2d7a')

# Labels and titles
#ax.set_ylabel('Normalized Prevalence')
#ax.set_title('Top-10 Most Common ADRs (X vs FDA)')
ax.set_xticks(x)
ax.set_xticklabels([adr.title() for adr in top_adrs], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate values
for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0.01:
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Save and display
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("Results/top10_adr_bar_combined.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# Metric Computation (IoU + ASD only)
# ======================================================

def get_heights_for_brand(brand, top_adrs, twitter_adr_counts, brand_level_fda, eps=1e-6):
    """Extract normalized h_twitter and h_fda vectors (avoid zeros)."""
    h_twitter = np.array(
        [twitter_adr_counts.get(brand, {}).get(a, 0.0) for a in top_adrs], float
    )
    h_fda = np.array(
        [brand_level_fda.get(brand, {}).get(a, 0.0) for a in top_adrs], float
    )

    # Avoid division errors
    h_twitter = np.maximum(h_twitter, eps)
    h_fda = np.maximum(h_fda, eps)

    # Normalize
    #h_twitter /= h_twitter.sum()
    #h_fda     /= h_fda.sum()
    return h_twitter, h_fda

def get_heights_for_brand_from_df(df_brand):
    h_twitter = df_brand.loc["X"].values.astype(float)
    h_fda     = df_brand.loc["FDA"].values.astype(float)
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


def compute_chisquare(h_obs, h_exp):
    """Chi-square statistic between observed (Twitter) and expected (FDA) distributions."""
    return np.sum((h_obs - h_exp) ** 2 / h_exp)

# -----------------------------
# Build brand dataframe (max/min + Twitter + FDA), normalized 0..1
# -----------------------------
def brand_df_for_origami_star(brand, axes, twitter_dict, fda_dict, eps=EPSILON, renorm=True):
    t = np.array([twitter_dict.get(brand, {}).get(a, 0.0) for a in axes], float)
    f = np.array([fda_dict.get(brand, {}).get(a, 0.0)     for a in axes], float)
    for arr in (t, f):
        arr[:] = np.maximum(arr, eps)
        s = arr.sum()
        if renorm and s > 0: arr[:] = arr / s
    df = pd.DataFrame([np.ones(len(axes)), np.zeros(len(axes)), t, f], columns=axes)
    df.index = ["max", "min", "X", "FDA"]
    return df

# -----------------------------
# 2×2 figure with three subplots + ONE legend
# -----------------------------
brands = ["Ozempic", "Wegovy", "Rybelsus"]
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
axs = axs.ravel()

pcols = [["#00a19a", "#ff2d7a"]]*3
#pfcols = [["#00a19a22", "#ff2d7a22"]]*3
pfcols = [["#00a19a11", "#ff2d7a11"]]*3  # extremely faint fill

pltys  = [[1, 2]]*3

shared_handles = shared_labels = None

#for i, brand in enumerate(brands):
panel_map = {
    "Ozempic": 0,
    "Wegovy": 1,
    "Rybelsus": 2   # <-- move Rybelsus to center
}

for brand in brands:
    i = panel_map[brand]


    df_brand = brand_df_for_origami_star(
        brand, top_adrs, twitter_adr_counts, brand_level_fda
    )

    # ---- Metrics for this brand ----
   

    h_twitter, h_fda = get_heights_for_brand_from_df(df_brand)
    iou = compute_iou(h_twitter, h_fda)
    asd_t_f, asd_f_t = compute_asd(h_twitter, h_fda)
    chi2 = compute_chisquare(h_twitter, h_fda)


    print(f"\n=== {brand} Metrics ===")
    print(f"IoU: {iou:.4f}")
    print(f"ASD(Twitter|FDA): {asd_t_f:.4f}")
    print(f"ASD(FDA|Twitter): {asd_f_t:.4f}")
    print(f"chi_square:{chi2:.4f}")

    '''ax, handles, labels = origami_star_plot(
        df_brand,
        seg=10, cglty=3, cglwd=1, cglcol="#cfcfcf",
        title=f"{brand}: FDA vs Twitter (Top-10 ADRs)",
        centerzero=False,
        vlabels=top_adrs,
        pcol=pcols[i], plty=pltys[i], plwd=[2, 2],
        pfcol=pfcols[i],
        ax=axs[i],
        return_handles=True,
        draw_spokes_from_center=True
    )

    if shared_handles is None:
        shared_handles, shared_labels = handles, labels'''
    
    ax, handles, labels = origami_star_plot(
        df_brand,
        seg=10, cglty=3, cglwd=1, cglcol="#cfcfcf",
        #title=f"{brand}: FDA vs Twitter (Top-10 ADRs)",
        title=f"{brand}: FDA vs X (Top-10 ADRs)\n(IoU = {iou:.3f}, ASD = {asd_t_f:.3f})",
        centerzero=False,
        vlabels=top_adrs,
        pcol=pcols[i], plty=pltys[i], plwd=[2, 2],
        pfcol=pfcols[i],
        ax=axs[i],
        return_handles=True,
        draw_spokes_from_center=True
)
    

# ⭐ Add this block for metrics
    '''frame = ax.get_xlim()[1]
    ax.text(
        frame * 0.80,
        -frame * 0.80,
        f"IoU = {iou:.3f}\nASD = {asd_t_f:.3f}",
        ha="left", va="top",
        fontsize=14,
        color="#333333",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )'''

if shared_handles is None:
    shared_handles, shared_labels = handles, labels



# shared legend in the 4th (empty) panel
axs[3].axis('off')
if shared_handles:
    axs[3].legend(shared_handles, shared_labels, loc='center',
                  frameon=True, framealpha=0.95, edgecolor="#DDDDDD")


fig.patch.set_facecolor("white")
plt.tight_layout()
pos = axs[2].get_position()
axs[2].set_position([pos.x0 + 0.18, pos.y0 - 0.1, pos.width, pos.height])
plt.savefig("Results/brand_level_origami.png",
            dpi=300, bbox_inches="tight")
plt.show()






