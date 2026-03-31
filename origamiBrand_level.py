# origami_star_2x2_from_df_fda.py
# Uses your df + FDA files to build Top-10 ADR axes, applies the
# original "origami" scaling formula from R, but renders each series
# as a star (k triangles) by inserting recess points between axes.

import warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------
# Display config
# ---------------------------------------
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
})

EPSILON = 1e-6

# ---------------------------------------
# 0) Origami geometry helpers
# ---------------------------------------
def _lty_map(r_lty):
    # R lty -> Matplotlib
    return {1:"-", 2:"--", 3:":", 4:"-.", 5:(0,(1,1)), 6:(0,(3,1,1,1))}.get(r_lty, "-")

def _origami_angles(n):
    # R: theta <- seq(90, 450, length=n+1)*pi/180; theta <- theta[1:n]
    theta = np.linspace(90, 450, n+1)[:-1] * np.pi/180.0
    return theta

def _interp_na_cartesian(j, vals_row, xx, yy, df_max, df_min, seg, CGap):
    """
    Reproduce the NA interpolation from R: compute left/right valid neighbors,
    then intersect the axis-j ray with the segment between scaled left/right points.
    """
    n = len(vals_row)
    # find left/right non-NaN
    left = (j-1) % n
    while np.isnan(vals_row[left]) and left != j:
        left = (left-1) % n
    right = (j+1) % n
    while np.isnan(vals_row[right]) and right != j:
        right = (right+1) % n

    # scale function per R formula for a single value v
    # r = CGap/(seg+CGap) + (v - min) / (max - min) * seg/(seg+CGap)
    denomL = (df_max[left] - df_min[left]) if (df_max[left] - df_min[left]) != 0 else 1.0
    denomR = (df_max[right]- df_min[right]) if (df_max[right]- df_min[right]) != 0 else 1.0

    rL = CGap/(seg+CGap) + ((vals_row[left] - df_min[left]) / denomL) * (seg/(seg+CGap))
    rR = CGap/(seg+CGap) + ((vals_row[right]- df_min[right]) / denomR) * (seg/(seg+CGap))

    xxleft, yyleft  = xx[left]*rL,  yy[left]*rL
    xxright, yyright= xx[right]*rR, yy[right]*rR

    # ensure xxleft <= xxright (to match R branch)
    if xxleft > xxright:
        xxleft, xxright = xxright, xxleft
        yyleft, yyright = yyright, yyleft

    # intersection of the line through origin along axis j with the
    # line segment between (xxleft,yyleft) and (xxright,yyright)
    # Derived from R’s algebra in the code.
    xj, yj = xx[j], yy[j]
    num = (yyleft * xxright - yyright * xxleft)
    den = (yj * (xxright - xxleft) - xj * (yyright - yyleft))
    if den == 0:
        # fallback to origin
        xj_sc, yj_sc = 0.0, 0.0
    else:
        xj_sc = xj * num / den
        yj_sc = (yj / xj) * xj_sc if xj != 0 else np.sign(yj) * abs(xj_sc)  # keep direction
    return xj_sc, yj_sc

def _scale_row_to_cartesian(vals_row, xx, yy, df_max, df_min, seg, CGap, na_itp=True):
    """
    For one data row (a series), compute the cartesian polygon points
    (xxs, yys) using the exact R scaling and NA interpolation logic.
    """
    n = len(vals_row)
    xxs = np.zeros(n)
    yys = np.zeros(n)
    # vectorized radii for non-NaN
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

def _to_star_path_from_xy(xxs, yys, inner_radius):
    """
    Convert the polygon defined by successive (xxs,yys) into a star by
    inserting a recess point halfway between each pair of vertices,
    with radius fixed to the inner ring (R's inner ring = CGap/(seg+CGap)).
    """
    n = len(xxs)
    star_x = []
    star_y = []
    # precompute angles of each vertex (tip)
    tip_thetas = np.arctan2(yys, xxs)
    for j in range(n):
        # tip
        star_x.append(xxs[j])
        star_y.append(yys[j])
        # recess halfway angle
        t_next = tip_thetas[(j+1) % n]
        t_curr = tip_thetas[j]
        # unwrap to avoid jumps near -pi/pi
        dt = (t_next - t_curr + np.pi) % (2*np.pi) - np.pi
        t_mid = t_curr + dt/2.0
        star_x.append(inner_radius*np.cos(t_mid))
        star_y.append(inner_radius*np.sin(t_mid))
    # close path
    star_x.append(star_x[0])
    star_y.append(star_y[0])
    return np.array(star_x), np.array(star_y)

def origami_star_plot(
    df,                 # DataFrame: row0=max, row1=min, row2..=series
    seg=4,
    cglty=3, cglwd=1, cglcol="navy",
    title="",
    na_itp=True,
    centerzero=False,
    vlabels=None, vlcex=None,
    pcol=None, plty=None, plwd=1, pfcol=None,
    inner_override=None,
    ax=None
):
    """
    Python port of R's origami_func *with the same scaling*,
    but render each series as a STAR (k triangles).

    df: first row = max, second row = min, rows 3.. = series to draw.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    n = df.shape[1]
    if n < 3:
        raise ValueError("The number of variables must be >= 3.")

    # Angles & unit directions (R ordering)
    theta = _origami_angles(n)
    xx, yy = np.cos(theta), np.sin(theta)

    # Center gap
    CGap = 0 if centerzero else 1

    # Prepare styles
    series = df.shape[0]
    SX = series - 2
    if pcol is None:  pcol  = ["C0", "C1", "C2", "C3"]
    if plty is None:  plty  = [1, 2, 1, 2]
    if pfcol is None: pfcol = [None]*SX
    pcol  = (pcol  * ((SX + len(pcol)-1)//len(pcol)))[:SX]
    plty  = (plty  * ((SX + len(plty)-1)//len(plty)))[:SX]
    pfcol = (pfcol * ((SX + len(pfcol)-1)//len(pfcol)))[:SX]

    # Setup axes
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111)
        created_fig = True
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    if title: ax.set_title(title, pad=12, fontsize=13, fontweight="bold")

    # Inner ring radius (R's "0" ring) and guide rings
    inner_r = CGap/(seg + CGap) if inner_override is None else inner_override
    # guide spokes (like R's arrows)
    for k in range(n):
        x0, y0 = xx[k]/(seg+CGap), yy[k]/(seg+CGap)
        ax.arrow(x0, y0, xx[k]-x0, yy[k]-y0, width=0.0, head_width=0.0,
                 linewidth=cglwd, linestyle=_lty_map(cglty), color=cglcol, length_includes_head=True, zorder=0)

    # axis labels
    labels = list(df.columns) if vlabels is None else list(vlabels)
    for k in range(n):
        ax.text(xx[k]*1.2, yy[k]*1.2, labels[k], ha="center", va="center",
                fontsize=(10 if vlcex is None else 10*vlcex))

    # Concentric polygons (guide rings)
    for i in range(1, seg+1):
        r = (i + CGap) / (seg + CGap)
        ax.plot(xx*r, yy*r, linestyle=_lty_map(cglty), linewidth=cglwd, color=cglcol, alpha=0.6, zorder=0)

    # Draw each data row as STAR
    df_max = df.iloc[0, :].astype(float).values
    df_min = df.iloc[1, :].astype(float).values

    for irow in range(2, series):
        vals = df.iloc[irow, :].astype(float).values
        # cartesian polygon points with origami scaling
        xxs, yys = _scale_row_to_cartesian(vals, xx, yy, df_max, df_min, seg, CGap, na_itp=na_itp)
        # convert to star path by inserting inner recesses
        sx, sy = _to_star_path_from_xy(xxs, yys, inner_r)
        # fill + outline
        face = pfcol[irow-2]
        edge = pcol[irow-2]
        ax.fill(sx, sy, color=face, alpha=0.25 if face is None else 1.0, zorder=2)
        ax.plot(sx, sy, color=edge, linewidth=(plwd if isinstance(plwd,(int,float)) else plwd[irow-2]),
                linestyle=_lty_map(plty[irow-2]), marker="o", markersize=3, zorder=3)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax

# ---------------------------------------
# 1) Load your TWITTER DF
# ---------------------------------------
df_tw = pd.read_pickle("Results/1adrs2_with_extractions.pkl")
df_tw["Extracted_ADRs"] = df_tw["Extracted_ADRs"].apply(lambda x: [i.lower() for i in x] if isinstance(x, list) else [])
df_tw["Brand"] = df_tw["Primary_Brand"].fillna('')

# ---------------------------------------
# 2) FDA files + loader (same as your code)
# ---------------------------------------
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
    total = dfx["Prevalence"].sum()
    return dict(zip(dfx["ADR"], dfx["Prevalence"] / total)) if total else {}

# brand-level FDA aggregation by brand (all doses)
brand_level_fda = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus"]:
    keys = [k for k in fda_info if k.startswith(brand)]
    acc = Counter()
    for k in keys:
        acc.update(load_fda(*fda_info[k]))
    tot = sum(acc.values())
    brand_level_fda[brand] = {k: v/tot for k, v in acc.items()} if tot else {}

# ---------------------------------------
# 3) Twitter normalized ADR shares per brand
# ---------------------------------------
twitter_adr_counts = {}
for brand in ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]:
    adrs = [adr for row in df_tw[df_tw["Brand"] == brand]["Extracted_ADRs"] for adr in row]
    cnt = Counter(adrs)
    tot = sum(cnt.values())
    twitter_adr_counts[brand] = {k: v/tot for k, v in cnt.items()} if tot else {}

# ---------------------------------------
# 4) Global Top-10 ADR axes (Twitter + FDA across all)
# ---------------------------------------
all_counters = list(twitter_adr_counts.values()) + [load_fda(*v) for v in fda_info.values()]
combined = Counter()
for c in all_counters:
    combined.update(c)
top_adrs = [adr for adr, _ in combined.most_common(10)]   # fixed axes for all subplots

# ---------------------------------------
# 5) Build df for a BRAND: max, min, twitter, fda (on same axes)
# ---------------------------------------
def brand_df_for_origami_star(brand, axes, twitter_dict, fda_dict, eps=EPSILON, renorm=True):
    # vectors aligned to axes
    t = np.array([twitter_dict.get(brand, {}).get(a, 0.0) for a in axes], float)
    f = np.array([fda_dict.get(brand, {}).get(a, 0.0)     for a in axes], float)
    # epsilon + renormalize (keeps shapes defined & comparable as distributions)
    for arr in (t, f):
        arr[:] = np.maximum(arr, eps)
        s = arr.sum()
        if renorm and s > 0: arr[:] = arr / s
    # dataframe expected by origami: row0=max, row1=min, row2..=series
    df = pd.DataFrame([np.ones(len(axes)), np.zeros(len(axes)), t, f], columns=axes)
    df.index = ["max", "min", "Twitter", "FDA"]
    return df

# ---------------------------------------
# 6) 2×2 figure with the requested 3 subplots
# ---------------------------------------
brands = ["Ozempic", "Wegovy", "Rybelsus"]  # order per your request
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.ravel()

colors = {
    "Twitter": {"edge":"#00a19a", "fill":"#00a19a30", "lty":1},
    "FDA":     {"edge":"#ff2d7a", "fill":"#ff2d7a30", "lty":2},
}

for i, brand in enumerate(brands):
    df_brand = brand_df_for_origami_star(brand, top_adrs, twitter_adr_counts, brand_level_fda)
    # Call the origami star plotter once per brand (it will draw both rows 3 & 4 as stars)
    origami_star_plot(
        df_brand,
        seg=4,                     # same default as R
        cglty=3, cglwd=1, cglcol="#b0b0b0",
        title=f"{brand}: FDA vs Twitter (Top-10 ADRs)",
        na_itp=True,
        centerzero=False,
        vlabels=top_adrs,
        pcol=[colors["Twitter"]["edge"], colors["FDA"]["edge"]],
        plty=[colors["Twitter"]["lty"],  colors["FDA"]["lty"]],
        plwd=[2,2],
        pfcol=[colors["Twitter"]["fill"], colors["FDA"]["fill"]],
        ax=axs[i],
    )

# Hide the 4th empty subplot
axs[3].set_visible(False)

fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("Results/brand_level_origami_star_2x2.png", dpi=300, bbox_inches="tight")
plt.show()
