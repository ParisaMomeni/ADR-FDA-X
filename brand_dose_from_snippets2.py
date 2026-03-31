# : for data/1adrs2.pkl did I ignore  empty cols of ADRs
#Which brand(s) are mentioned (Wegovy / Ozempic / Rybelsus / Unknown)

#What mg doses appear near brand mentions (e.g., 0.5 mg, 1 mg, 2.4 mg)

#Whether the dose matches canonical escalation schedules Whether it's off-label Whether multiple brands appear in the same post
#Save all of this in a new dataframe
#This pipeline does NOT use ADRs at all.

import pandas as pd
import re
import json

# ----------------------
# Canonical doses & dose buckets per brand
# ----------------------

CANONICAL = {
    "Wegovy":  {"keep": [2.4],       "other_ok": [0.25, 0.5, 1.0, 1.7]},
    "Ozempic": {"keep": [0.5, 1.0],  "other_ok": []},
    "Rybelsus":{"keep": [7.0, 14.0], "other_ok": [3.0]},
}
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
OTHER_LABEL = "Other"
OFF_LABEL     = "Off-label"
UNKNOWN_BRAND = "Unknown"
UNKNOWN_BUCKET = "Unknown"

ALNUM = r"A-Za-z0-9"

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
BRANDS = list(CANONICAL.keys())
 
# mg value (avoid mg/dL etc.)
#MG_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*mg(?!\s*/?\s*d[lL])", re.I)

# Accept:
# - No space: 0.5mg, 1mg
# - Spaces/tabs/Unicode spaces (incl. NBSP, thin/figure, narrow no-break, etc.) up to 3 on each side of an optional "~" or "≈"
# - European decimal comma: 1,0 mg
# - Optional "~" or "≈" before mg (with or without surrounding spaces)
# Exclude:
# - mg/dL, mg/L (with optional spaces and slash)
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

    keep = CANONICAL[brand]["keep"]
    other_ok = CANONICAL[brand]["other_ok"]

    # Choose the mg closest to any canonical value (keep or other_ok)
    # Finds the closest extracted dose to any known dose.
    # # Only snaps to a canonical "keep" dose if it's very close (±0.15 mg). What if I want to return other_ok doses?
    best = None
    best_dist = float("inf")
    for v in mg_vals:             # v = each extracted dose from text
        for k in keep + other_ok: # k = each known dose for this brand
            d = abs(v - k)        # compute distance between v and k
            if d < best_dist:     
                best_dist = d
                best = v

    # Tolerance to snap to canonical doses
    TOL = 0.15
    
    # Check keep buckets first
    for k in keep:
        if best is not None and abs(best - k) <= TOL:
            return f"{k:g} mg"
    return OFF_LABEL
'''
    #What if I want to return other_ok doses?
    # First try snapping to keep doses
    for k in keep:
        if best is not None and abs(best - k) <= TOL:
            return f"{k:g} mg"  # Canonical target dose

    # Then try snapping to other_ok doses
    for k in other_ok:
        if best is not None and abs(best - k) <= TOL:
            return f"{k:g} mg (other_ok)"  # Step-up or acceptable but not primary

    # If no match
    return OFF_LABEL
'''


def summarize_brand_dose_table(df: pd.DataFrame, title_col="Title", text_col="Snippet"):
    # Build the searchable text (Title + Snippet if Title exists)
    if title_col and title_col in df.columns:
        full_text = (df[title_col].fillna('') + ' ' + df[text_col].fillna('')).astype(str)
    else:
        full_text = df[text_col].fillna('').astype(str)

    rows = []
    for txt in full_text:
        brand, pos = detect_brand_and_pos(txt)
        mg_vals = find_mg_near(txt, pos, window=60)
        bucket = dose_bucket(brand, mg_vals)

        # Map buckets exactly as requested
        if brand == "Wegovy":
            bucket = bucket if bucket in {"2.4 mg", OFF_LABEL} else OTHER_LABEL
        elif brand == "Ozempic":
            bucket = bucket if bucket in {"0.5 mg", "1 mg", OFF_LABEL} else OTHER_LABEL
        elif brand == "Rybelsus":
            bucket = bucket if bucket in {"7 mg", "14 mg", OFF_LABEL} else OTHER_LABEL
        else:
            brand, bucket = UNKNOWN_BRAND, UNKNOWN_BUCKET


        rows.append((brand, bucket))

    grouped = (
        pd.DataFrame(rows, columns=["Brand", "Dose_Bucket"])
          .groupby(["Brand", "Dose_Bucket"], dropna=False)
          .size()
          .reset_index(name="Count")
    )

    # Overall prevalence (safe if total_all == 0)
    total_all = grouped["Count"].sum()
    grouped["Prevalence_All(%)"] = (grouped["Count"] / total_all * 100).round(2) if total_all else 0.0
    # brand-level totals & prevalence (dose-agnostic), repeated for each row of that brand
    grouped["Brand_Count"] = grouped.groupby("Brand")["Count"].transform("sum")
    grouped["Brand_Prevalence_All(%)"] = (
    grouped["Brand_Count"] / total_all * 100
).round(2) if total_all else 0.0

    # ✅ FIX: use transform so the index aligns with `grouped`
    grouped["Prevalence_WithinBrand(%)"] = (
        grouped["Count"] / grouped.groupby("Brand")["Count"].transform("sum") * 100
    ).round(2)

    # Nice ordering without per-row categorical assignment
    brand_order = ["Wegovy", "Ozempic", "Rybelsus", "Unknown"]
    dose_order_map = {
        "Wegovy":   {"2.4 mg": 0, OFF_LABEL: 1, OTHER_LABEL: 2},
        "Ozempic":  {"0.5 mg": 0, "1 mg": 1, OFF_LABEL: 2, OTHER_LABEL: 3},
        "Rybelsus": {"7 mg": 0, "14 mg": 1, OFF_LABEL: 2, OTHER_LABEL: 3},
        "Unknown":  {UNKNOWN_BUCKET: 0},
    }

    grouped["Brand"] = pd.Categorical(grouped["Brand"], categories=brand_order, ordered=True)
    grouped["Dose_Order"] = grouped.apply(
        lambda r: dose_order_map.get(str(r["Brand"]), {}).get(r["Dose_Bucket"], 99),
        axis=1
    )

    grouped = grouped.sort_values(["Brand", "Dose_Order", "Count"], ascending=[True, True, False]).drop(columns=["Dose_Order"]).reset_index(drop=True)
    return grouped

# ----------------------
#Save brand/dose extractions to DataFrame
# ----------------------
# NEW: get all brand hits, not just the first
def detect_all_brands_and_positions(text: str):
    txt = str(text or "")
    hits = []
    for brand, pat in BRAND_PATTERNS.items():
        for m in pat.finditer(txt):
            hits.append((brand, m.start()))
    hits.sort(key=lambda x: x[1])
    return hits  # list[(brand, pos)]

# NEW: mg search around multiple positions
def find_mg_near_positions(text: str, positions: list[int], window: int = 60) -> list[float]:
    txt = str(text or "")
    vals = []
    for pos in positions:
        ctx = txt[max(0, pos-window): min(len(txt), pos+window)]
        for m in MG_RE.finditer(ctx):
            try:
                vals.append(float(m.group(1).replace(",", ".")))
            except:
                pass
    return vals

def annotate_extractions(df: pd.DataFrame, title_col="Title", text_col="Snippet", window: int = 60) -> pd.DataFrame:
    
    if title_col and title_col in df.columns:
        full_text = (df[title_col].fillna('') + ' ' + df[text_col].fillna('')).astype(str)
    else:
        full_text = df[text_col].fillna('').astype(str)

    brands_col, brands_str_col = [], []
    primary_brand_col = []
    doses_all_col, doses_all_str_col = [], []
    doses_primary_col, doses_primary_str_col = [], []
    dose_bucket_col, off_label_col, has_dose_col = [], [], []
    brand_to_doses_json_col = []

    for txt in full_text:
        # all brand hits
        hits = detect_all_brands_and_positions(txt)
        brands = sorted(set([b for b, _ in hits]), key=lambda b: [p for (bb,p) in hits if bb==b][0]) if hits else [UNKNOWN_BRAND]
        primary = brands[0] if brands else UNKNOWN_BRAND

        # per-brand positions
        brand_positions = {}
        for b, pos in hits:
            brand_positions.setdefault(b, []).append(pos)

        # per-brand doses (near that brand's positions)
        brand_to_doses = {}
        for b, poss in brand_positions.items():
            brand_to_doses[b] = find_mg_near_positions(txt, poss, window=window)

        # union across brands
        doses_union = sorted({v for vs in brand_to_doses.values() for v in vs})

        # doses near primary brand
        doses_primary = brand_to_doses.get(primary, []) if primary in BRANDS else []

        # bucket for primary brand
        bucket = dose_bucket(primary, doses_primary)

        # flags
        off_label = (bucket == OFF_LABEL)
        has_dose = bool(doses_union)

        # pretty strings for CSV
        brands_str = "|".join(brands)
        doses_all_str = "; ".join(f"{v:g} mg" for v in doses_union)
        doses_primary_str = "; ".join(f"{v:g} mg" for v in sorted(set(doses_primary)))

        # push
        brands_col.append(brands)
        brands_str_col.append(brands_str)
        primary_brand_col.append(primary)
        doses_all_col.append(doses_union)
        doses_all_str_col.append(doses_all_str)
        doses_primary_col.append(doses_primary)
        doses_primary_str_col.append(doses_primary_str)
        dose_bucket_col.append(bucket)
        off_label_col.append(off_label)
        has_dose_col.append(has_dose)
        brand_to_doses_json_col.append(json.dumps(brand_to_doses))  # portable

    df = df.copy()
    df["Brands"] = brands_col                # list
    df["Brands_Str"] = brands_str_col        # "Wegovy|Ozempic"
    df["Primary_Brand"] = primary_brand_col
    df["Doses_All_mg"] = doses_all_col       # list[float]
    df["Doses_All_Str"] = doses_all_str_col  # "0.5 mg; 1 mg"
    df["Doses_Near_Primary_mg"] = doses_primary_col
    df["Doses_Near_Primary_Str"] = doses_primary_str_col
    df["Dose_Bucket"] = dose_bucket_col      # keep/Off-label/Other/Unknown
    df["Off_Label"] = off_label_col          # bool
    df["Has_Dose"] = has_dose_col            # bool
    df["Brand_to_Doses_JSON"] = brand_to_doses_json_col
    return df



#-  ----------------------
# Group by Author, Primary_Brand, and Dose_Bucket


# ----------------------
# Run on your data

# ----------------------
if __name__ == "__main__":
    #df= pd.read_pickle("data/1adrs2.pkl")
    #df = pd.read_excel("data/processed_data_adrs_threshold0.5.xlsx")  # expects columns like Title, Snippet
    df = pd.read_excel("data/processed_data_merged_noRT_adrs_threshold0.55.xlsx")
    table = summarize_brand_dose_table(df, title_col="Title" if "Title" in df.columns else None, text_col="Snippet")
    # 1) Annotate rows with brands/doses
    df_annot = annotate_extractions(df, title_col="Title" if "Title" in df.columns else None, text_col="Snippet", window=60)
    # Filter only Rybelsus rows and count the nearby doses
    rybelsus_dose_counts = (
        df_annot[df_annot["Primary_Brand"] == "Rybelsus"]
        .explode("Doses_Near_Primary_mg")
        .dropna(subset=["Doses_Near_Primary_mg"])
        .groupby("Doses_Near_Primary_mg")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
)

    print(rybelsus_dose_counts)

#-------------
    
    rybelsus_doses = df_annot[
        (df_annot["Primary_Brand"] == "Rybelsus")
    ]

    # Expand and count all individual dosage mentions (flattened)
    rybelsus_dose_counts = (
        rybelsus_doses.explode("Doses_Near_Primary_mg")
        .dropna(subset=["Doses_Near_Primary_mg"])
        .groupby("Doses_Near_Primary_mg")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    print(rybelsus_dose_counts)

#--------------


    # Save the enriched table (now includes the new columns)
    df_annot.to_pickle("Results/1adrs2_with_extractions.pkl")
    df_annot.to_csv("Results/1adrs2_with_extractions.csv", index=False)
    print(table)
    table.to_csv("Results/brand_dose_prevalence_table.csv", index=False)
    table = summarize_brand_dose_table(df, title_col="Title" if "Title" in df.columns else None, text_col="Snippet")

  