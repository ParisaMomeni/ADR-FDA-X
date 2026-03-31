import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


BRANDS = ["Wegovy", "Ozempic", "Rybelsus"]

brand_aliases = {
    "Wegovy": [
        "wegovy", "weg", "wegvy", "wgvy", "wegovi", "weggo", "#wegovy", "#weg", "#wegvy"
    ],
    "Ozempic": [
        "ozempic", "oz", "ozz", "ozi", "ozmpic", "ozmpik", "zempic", "ozzy", "#ozempic", "#oz", "#ozzy"
    ],
    "Rybelsus": [
        "rybelsus", "ryb", "rybselus", "rybslus", "rybby", "rybel", "#rybelsus", "#ryb"
    ]
}
UNKNOWN_BRAND = "Unknown"
ALNUM = r"A-Za-z0-9"
OTHER_LABEL = "Other"
UNKNOWN_BUCKET = "Unknown"

def compile_brand_patterns(brand_aliases: dict[str, list[str]]) -> dict[str, re.Pattern]:
    patterns = {}
    for brand, aliases in brand_aliases.items():
        # normalize & escape aliases
        alts = "|".join(re.escape(a.lstrip("#").lower()) for a in aliases)
        # optional '#' before alias; custom boundaries allow punctuation around it
        pat = rf"(?<![{ALNUM}])#?(?:{alts})(?![{ALNUM}])"
        patterns[brand] = re.compile(pat, re.I)
    return patterns
BRAND_PATTERNS = compile_brand_patterns(brand_aliases)

def extract_brand(text):
    text = str(text).lower()
    for brand, pattern in BRAND_PATTERNS.items():
        if pattern.search(text):
            return brand
    return UNKNOWN_BRAND

MG_RE = re.compile(
    r"""(?<![A-Za-z0-9])                             # don't start in the middle of a word/number
        (\d+(?:[.,]\d+)?)                             # number (supports 1,0 or 1.0)
        (?:                                           # optional spacing/approx block
            (?:[\s\u00A0\u2000-\u200A\u202F\u205F\u3000]){0,3}
            [~≈]?
            (?:[\s\u00A0\u2000-\u200A\u202F\u205F\u3000]){0,3}
        )?
        mg                                           # unit
        (?!\s*/?\s*d?l)                              # not mg/dL or mg/L
    """,
    re.I | re.X
)





#---------------- 


def detect_brand_and_pos(text: str):
    txt = str(text or "")
    hits = []
    for b, pat in BRAND_PATTERNS.items():
        m = pat.search(txt)
        if m:
            hits.append((b, m.start()))
    if not hits:
        return UNKNOWN_BRAND, None
    hits.sort(key=lambda x: x[1])
    return hits[0]  # (brand, pos)

def find_mg_near(text: str, center: int | None, window: int = 60):
    txt = str(text or "")
    ctx = txt if center is None else txt[max(0, center-window): min(len(txt), center+window)]
    vals = []
    for m in MG_RE.finditer(ctx):
        try:
            vals.append(float(m.group(1).replace(",", ".")))
        except:
            pass
    return vals

def dose_bucket(brand: str, mg_vals: list[float]) -> str:
    if brand not in BRANDS:
        return UNKNOWN_BUCKET
    if not mg_vals:
        return OTHER_LABEL
    return "; ".join(f"{v:.1f} mg" for v in mg_vals)

def summarize_primary_brand_dose(df: pd.DataFrame, title_col="Title", text_col="Snippet"):
    # Build the searchable text (Title + Snippet if Title exists)
    if title_col and title_col in df.columns:
        full_text = (df[title_col].fillna('') + ' ' + df[text_col].fillna('')).astype(str)
    else:
        full_text = df[text_col].fillna('').astype(str)

    rows = []
    for txt in full_text:
        brand, pos = detect_brand_and_pos(txt)
        if brand == "Rybelsus":
            mg_vals = find_mg_near(txt, pos, window=60)
            bucket = dose_bucket(brand, mg_vals)

            rows.append((brand, bucket))

    grouped = (
        pd.DataFrame(rows, columns=["Primary_Brand", "Primary_Dose"])
          .groupby(["Primary_Brand", "Primary_Dose"], dropna=False)
          .size()
          .reset_index(name="Count")
    )

    rybelsus_grouped = grouped[grouped["Primary_Brand"] == "Rybelsus"].copy()
    rybelsus_grouped = rybelsus_grouped[rybelsus_grouped["Primary_Dose"] != OTHER_LABEL]
    return rybelsus_grouped

#------------------

if __name__ == "__main__":
    

    df = pd.read_pickle("data/1adrs2.pkl")
    snippets = df["Snippet"].tolist()

    summary_primary = summarize_primary_brand_dose(df)

    print(summary_primary)

    summary_primary.to_csv("Results/primary_brand_dose_summary.csv", index=False)

    
#----------------------- test:
# Save to CSV
    rybelsus_rows = summary_primary[summary_primary["Primary_Brand"] == "Rybelsus"]
    rybelsus_rows.to_csv("Results/1adrs2_rybelsus_only.csv", index=False)

    print("✅ Saved Rybelsus-related rows to Results/1adrs2_rybelsus_only.csv")
#-------------------------
#----------------------- histogram chart:

'''# --- Create a histogram of Rybelsus dosage counts ---
def plot_dose_histogram(summary_df, brand="Rybelsus"):
    # Filter for the specified brand
    df_filtered = summary_df[summary_df["Primary_Brand"] == brand].copy()
       # Sort by Count (descending)
    df_filtered = df_filtered.sort_values("Count", ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_filtered["Primary_Dose"], df_filtered["Count"], color="skyblue", edgecolor="black")
    
    # Add counts as labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 10, f'{height}', ha='center', va='bottom', fontsize=8)

    plt.title(f"Distribution of Reported Dosages for {brand}", fontsize=14)
    plt.xlabel("Dosage (mg)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save or show
    plt.savefig(f"Results/{brand.lower()}_dose_histogram.png", dpi=300)
    plt.show()

# Call the plotting function
plot_dose_histogram(summary_primary)'''


# rybelsus_histogram.py
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# 1) Data (dose -> count)
# ----------------------------
raw = {
    "1.0 mg": 2,
    "1.5 mg": 1,
    "10.0 mg": 1,
    "14.0 mg": 119,
    "7.0 mg": 37,
    "2.0 mg": 1,
    "25.0 mg": 3,
    "3.0 mg": 41,
    "5.0 mg": 1,
    "50.0 mg": 27,
}

# Parse numeric values so the x-axis is in the correct numerical order
def dose_value(label: str) -> float:
    # expects strings like "14.0 mg"
    return float(label.lower().replace("mg", "").strip())

labels_sorted = sorted(raw.keys(), key=dose_value)
counts_sorted = [raw[k] for k in labels_sorted]

# ----------------------------
# 2) Typography configuration
# ----------------------------
# Try to use LaTeX fonts for a seamless match with your Overleaf document.
# If LaTeX isn't available locally, we fall back to Computer Modern-like serif.
'''USE_TEX = True
try:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.unicode_minus": False,  # safer when using TeX
    })
    # If you want to export a PGF for Overleaf, uncomment the next two lines:
    # mpl.use("pgf")
    # mpl.rcParams.update({"pgf.texsystem": "pdflatex"})
except Exception:
    # Fallback (very similar look without requiring LaTeX)
    USE_TEX = False
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })
'''
# ----------------------------
# 3) Plot
# ----------------------------
plt.figure(figsize=(10, 5))  # clean single figure

bars = plt.bar(labels_sorted, counts_sorted, edgecolor="black", linewidth=0.8)

# Label counts above bars (centered, small offset)
for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(1, height * 0.02),
            f"{int(height)}",
            ha="center", va="bottom", fontsize=11, fontweight='bold'  # bigger + bold counts
        )

# Title (bigger & bolder)
#title_txt = r"Distribution of Manually Recorded \textit{Rybelsus} Dosages" if USE_TEX \
   #         else "Distribution of Manually Recorded Rybelsus Dosages"
#plt.title(title_txt, fontsize=18, fontweight='bold')

# Axis labels (bigger & bolder)
plt.xlabel("Dosage (mg)", fontsize=16, fontweight='bold')
plt.ylabel("Count", fontsize=16, fontweight='bold')

# Ticks and grid tuned for print
plt.xticks(rotation=45, ha="right", fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.8)

plt.tight_layout()

# Save outputs
plt.savefig("rybelsus_dose_histogram.png", dpi=300)

try:
    plt.savefig("rybelsus_dose_histogram.pgf")  # For Overleaf (if LaTeX installed)
except Exception:
    pass

plt.show()
