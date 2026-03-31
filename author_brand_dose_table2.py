import pandas as pd
import json
import orjson
import re

# ---------- constants ----------
FDA_DOSES = {
    "Ozempic":  {"0.5 mg", "1 mg"},
    "Wegovy":   {"2.4 mg"},
    "Rybelsus": {"7 mg", "14 mg"},
}

# ---------- load ----------
df = pd.read_csv("Results/1adrs2_with_extractions.csv", low_memory=False)

# make sure Author is string
df["Author"] = df["Author"].astype(str)

# parse JSON column safely


def quick_parse(x):
    if not isinstance(x, str) or not x.strip() or x == "{}":
        return {}
    # find brand names
    brands = re.findall(r"'([^']+)': \[([^\]]*)\]", x)
    out = {}
    for brand, doses in brands:
        dose_list = [d.strip().strip("'\"") for d in doses.split(",") if d.strip()]
        out[brand] = dose_list
    return out

df["parsed"] = df["Brand_to_Doses_JSON"].map(quick_parse)

# ---------- group by author ----------
def combine_dicts(dicts):
    out = {}
    for d in dicts:
        for brand, doses in d.items():
            out.setdefault(brand, set()).update(doses)
    # convert sets to sorted lists for readability
    return {b: sorted(list(v)) for b, v in out.items()}

authors = (
    df.groupby("Author")["parsed"]
      .apply(combine_dicts)
      .reset_index()
)

# ---------- derive flags ----------
def author_flags(brand_to_doses):
    brands = {b for b in brand_to_doses if b not in ["Unknown", "[]", "", None]}
    doses  = {d for ds in brand_to_doses.values() for d in ds}
    
    any_brand   = len(brands) > 0
    n_brands    = len(brands)
    n_doses     = len(doses)
    
    # FDA-correct pairs
    keep_pairs = {
        (b, d) for b, ds in brand_to_doses.items()
               for d in ds if b in FDA_DOSES and d in FDA_DOSES[b]
    }
    n_keep = len(keep_pairs)
    
    any_offlabel = any(
        b in FDA_DOSES and d not in FDA_DOSES[b]
        for b, ds in brand_to_doses.items() for d in ds
    )
    
    return pd.Series({
        "any_brand": any_brand,
        "n_brands": n_brands,
        "n_doses": n_doses,
        "n_keep_pairs": n_keep,
        "any_offlabel": any_offlabel,
    })

author_flags_df = authors["parsed"].apply(author_flags)
authors = pd.concat([authors, author_flags_df], axis=1)

# ---------- percentages ----------
N_total = len(authors)
pct = lambda x: round(100.0 * x / N_total, 2) if N_total else 0.0

summary = {
    "Total unique authors": N_total,
    "Authors mentioning ≥1 brand": f"{authors['any_brand'].sum()} ({pct(authors['any_brand'].sum())}%)",
    "Authors mentioning >1 brand": f"{(authors['n_brands'] > 1).sum()} ({pct((authors['n_brands'] > 1).sum())}%)",
    "Authors mentioning any off-label dose": f"{authors['any_offlabel'].sum()} ({pct(authors['any_offlabel'].sum())}%)",
    "Authors mentioning exactly 1 brand & 1 FDA dose (no off-label)": f"{((authors['n_brands']==1)&(authors['n_keep_pairs']==1)&(~authors['any_offlabel'])).sum()} ({pct(((authors['n_brands']==1)&(authors['n_keep_pairs']==1)&(~authors['any_offlabel'])).sum())}%)"
}

# ---------- save ----------
authors.to_csv("Results/authors_grouped_simple.csv", index=False)
pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_csv("Results/author_summary_simple.csv", index=False)

print("Done. Wrote Results/authors_grouped_simple.csv and Results/author_summary_simple.csv")
