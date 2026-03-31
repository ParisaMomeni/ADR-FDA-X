#to do why the result for >= 1 and >0 are different brand 
import pandas as pd

# ---------- constants (match pipeline) ----------
OTHER_LABEL    = "Other"
OFF_LABEL      = "Off-label"
UNKNOWN_BRAND  = "Unknown"
UNKNOWN_BUCKET = "Unknown"

KEEP_DOSES_BY_BRAND = {
    "Wegovy":   {"2.4 mg"},
    "Ozempic":  {"0.5 mg", "1 mg"},
    "Rybelsus": {"7 mg", "14 mg"},
}

KNOWN_BRANDS = {"Ozempic", "Wegovy", "Rybelsus"}

# ---------- load ----------
df = pd.read_csv("Results/1adrs2_with_extractions.csv", low_memory=False)
author_col = "Author"
brand_col  = "Primary_Brand" 
bucket_col = "Dose_Bucket" 
if brand_col not in df.columns or bucket_col not in df.columns:
    raise ValueError(f"Expected columns '{brand_col}' and '{bucket_col}' in the CSV.")

cols_needed = [author_col, brand_col, bucket_col]
if "Brands_Str" in df.columns:
    cols_needed.append("Brands_Str")

acct_col = "Account Type"  
if acct_col in df.columns:
    cols_needed.append(acct_col)


posts = df[cols_needed].copy()
#posts[acct_col] = posts[acct_col].astype(str).str.strip().str.lower() 
posts[brand_col]  = posts[brand_col].astype(str).fillna("")
posts[bucket_col] = posts[bucket_col].astype(str).fillna("")
posts[author_col] = posts[author_col].astype(str).fillna("")
posts["has_brand"] = posts[brand_col].isin(KNOWN_BRANDS)
posts["has_dose"]  = ~posts[bucket_col].isin([OTHER_LABEL, UNKNOWN_BUCKET, "", "nan"])  # keep or off-label
posts["is_keep"] = False
for b, keep_set in KEEP_DOSES_BY_BRAND.items():
    posts.loc[(posts[brand_col] == b) & (posts[bucket_col].isin(keep_set)), "is_keep"] = True
posts["is_offlabel"] = posts[bucket_col].eq(OFF_LABEL)

# ---------- author-level aggregates ----------
num_unique_authors = posts[author_col].nunique()
#print("Number of unique authors in posts:", num_unique_authors) #Number of unique authors in posts: 436551
known_brand_posts = posts[posts[brand_col].isin(KNOWN_BRANDS)].copy()
unknown_brand_posts = posts[posts[brand_col] == UNKNOWN_BRAND].copy()
num_authors_known = known_brand_posts[author_col].nunique()
#print("Authors mentioning ≥ 1 brand", num_authors_known) #Authors mentioning ≥ 1 brand 402126
num_authors_unknown = unknown_brand_posts[author_col].nunique()
#print("Authors mentioning No brand", num_authors_unknown) #Authors mentioning No brand 61027

abd = (
    posts.groupby([author_col, brand_col, bucket_col], dropna=False)
       .size()
       .reset_index(name="posts")
)

ab = (
    posts[ [author_col, brand_col]]
       .drop_duplicates()   
       .groupby(author_col, sort=False)[brand_col]
       .agg(set)
)
# per-author set of KNOWN_BRANDS only
ab_known = (
    posts.loc[posts["has_brand"], [author_col, brand_col]]
         .drop_duplicates()
         .groupby(author_col, sort=False)[brand_col]
         .agg(set)
)

has_known_by_author = (
    posts.loc[posts[brand_col].isin(KNOWN_BRANDS), [author_col]]
         .drop_duplicates()
         .assign(has_known=True)
         .set_index(author_col)
)
has_unknown_by_author = (
    posts.loc[posts[brand_col].eq(UNKNOWN_BRAND), [author_col]]
         .drop_duplicates()
         .assign(has_unknown=True)
         .set_index(author_col)
)
authors_known_and_unknown_idx = has_known_by_author.join(has_unknown_by_author, how="inner").index
n_known_and_unknown = len(authors_known_and_unknown_idx)


#---------------------------------------
#---------------------------------------
keep_rows = (
    posts.loc[posts["is_keep"], [author_col, brand_col, bucket_col]]
       .drop_duplicates()
)
 #builds a set of (brand, dose) pairs per author. So keep_rows contains: author_col, brand_col, and bucket_col
keep_per_author = (
    keep_rows
    .groupby(author_col, sort=False)[[brand_col, bucket_col]]
    .apply(lambda g: set(map(tuple, g[[brand_col, bucket_col]].itertuples(index=False, name=None))))
)

offlabel_by_author = (
    posts.loc[posts["is_offlabel"], [author_col]]
       .drop_duplicates()
       .set_index(author_col)
       .assign(any_offlabel=True)
)

dose_by_author = (
    posts.loc[posts["has_dose"], [author_col]]
       .drop_duplicates()
       .set_index(author_col)
       .assign(any_dose=True)
)
print("Number of authors with any dose:", (dose_by_author))
print("Number of authors with any dose:", int(dose_by_author["any_dose"].sum()))

# Has >=1 brand flag per author
brand_by_author = (
    posts.loc[posts["has_brand"], [author_col]]
       .drop_duplicates()
       .set_index(author_col)
       .assign(any_brand=True)
)
authors = pd.DataFrame(index=pd.Index(posts[author_col].unique(), name=author_col)).sort_index()
authors = authors.join(brand_by_author, how="left").join(dose_by_author, how="left").join(offlabel_by_author, how="left")
authors["any_brand"]    = authors["any_brand"].fillna(False).astype(bool)
num_true = authors["any_brand"].sum()
num_true = authors["any_brand"].value_counts().get(True, 0)
authors["any_dose"]     = authors["any_dose"].fillna(False).astype(bool)
authors["any_offlabel"] = authors["any_offlabel"].fillna(False).astype(bool)
#authors["brands_set"] = authors.index.to_series().map(ab).where(lambda x: x.notna(), other=[set()]*len(authors))
authors["brands_set"] = authors.index.to_series().map(lambda a: ab_known.get(a, set()))
authors["n_brands"]     = authors["brands_set"].apply(len) 
authors["keep_pairs"]   = authors.index.to_series().map(keep_per_author).where(lambda x: x.notna(), other=[set()]*len(authors))
authors["n_keep_pairs"] = authors["keep_pairs"].apply(len)
# ---------- metrics ----------
N_total = int(len(authors))
pct = (lambda x: round(100.0 * x / N_total, 2) if N_total else 0.0)
N_any_brand = int(authors["any_brand"].sum())
print("N_any_brand:", N_any_brand) #N_total: 436551
#print("N_any_brand:", N_any_brand) #N_any_brand: 402126
N_multi_brand = int((authors["n_brands"] > 1).sum())
N_one_brand   = int((authors["n_brands"] == 1).sum())
N_one_brand_one_keep = int(((authors["n_brands"] == 1) & (authors["n_keep_pairs"] == 1) & (~authors["any_offlabel"])).sum())
N_any_offlabel = int(authors["any_offlabel"].sum())
N_both_brand_and_dose = int((authors["any_brand"] & authors["any_dose"]).sum())
N_brand_only = int((authors["any_brand"] & ~authors["any_dose"]).sum())
N_no_brand = (N_total) - (N_any_brand) #authors who never mention any known brand 34425
N_no_brand2 = num_authors_unknown #authors who have at least one post where
N_brand  = int((authors["n_brands"]>=0).sum()) #478770


per_brand_unique_authors = (
    posts.loc[posts["has_brand"], [author_col, brand_col]]
       .drop_duplicates()
       .groupby(brand_col, sort=False)[author_col]
       .nunique()
)

per_brand_unique_authors = per_brand_unique_authors.drop(labels=[UNKNOWN_BRAND, "", "nan", "None"], errors="ignore")
per_brand_unique_authors = per_brand_unique_authors.sort_values(ascending=False)

top_brand = per_brand_unique_authors.idxmax() if len(per_brand_unique_authors) else None
top_brand_authors = int(per_brand_unique_authors.max()) if len(per_brand_unique_authors) else 0

# ---------- print summary ----------
print("\n=== Author-level summary ===")
print(f"Total unique authors: {N_total}")

print(f"Authors mentioning at least 1 brand: {N_any_brand}  ({pct(N_any_brand)}%)")
print(f"Authors mentioning more than 1 brand: {N_multi_brand}  ({pct(N_multi_brand)}%)")
print(f"Authors mentioning exactly 1 brand & exactly 1 FDA-approved dose (no off-label): {N_one_brand_one_keep}  ({pct(N_one_brand_one_keep)}%)")
print(f"Authors mentioning any off-label dose: {N_any_offlabel}  ({pct(N_any_offlabel)}%)")
print(f"Authors mentioning both brand & (any) dose: {N_both_brand_and_dose}  ({pct(N_both_brand_and_dose)}%)")
print(f"Authors mentioning brand only (no dose): {N_brand_only}  ({pct(N_brand_only)}%)")
print(f"Authors never mentioning any brand: {N_no_brand}  ({pct(N_no_brand)}%)")
print(f"Authors mentioning both FDA-approved brand and no brand in different posts: {n_known_and_unknown}  ({pct(n_known_and_unknown)}%)")
if top_brand:
    print(f"Most-mentioned brand by unique authors: {top_brand}  ({top_brand_authors} authors, {pct(top_brand_authors)}% of all authors)")
else:
    print("Most-mentioned brand by unique authors: N/A")

print("\nPer-brand unique authors (any mention):")
print(per_brand_unique_authors)

# ---------- optional: save to CSV (summary metrics) ----------
summary = pd.DataFrame({
    "metric": [
        "Total unique authors",
        "Authors mentioning ≥1 brand",
        "Authors mentioning >1 brand",
        "Authors mentioning exactly 1 brand & exactly 1 FDA-approved dose (no off-label)",
        "Authors mentioning any off-label dose",
        "Authors mentioning both brand & dose",
        "Authors mentioning brand only (no dose)",
        "Authors never mentioning any brand",
        "Top brand by unique authors",
    ],
    "value": [
        N_total,
        f"{N_any_brand} ({pct(N_any_brand)}%)",
        f"{N_multi_brand} ({pct(N_multi_brand)}%)",
        f"{N_one_brand_one_keep} ({pct(N_one_brand_one_keep)}%)",
        f"{N_any_offlabel} ({pct(N_any_offlabel)}%)",
        f"{N_both_brand_and_dose} ({pct(N_both_brand_and_dose)}%)",
        f"{N_brand_only} ({pct(N_brand_only)}%)",
        f"{N_no_brand} ({pct(N_no_brand)}%)",
        f"{top_brand} ({top_brand_authors}, {pct(top_brand_authors)}%)" if top_brand else "N/A",
    ]
})
summary.to_csv("Results/author_level_summary_metrics.csv", index=False)

# ---------- per-brand: authors vs posts ----------
brand_for_posts = "Primary_Brand"
author_for_posts = author_col

total_authors_global = df[author_for_posts].astype(str).fillna("").nunique()
total_posts_global   = len(df)

brand_stats = (
    df.assign(_brand=df[brand_for_posts].astype(str).fillna(""))
      .groupby("_brand", dropna=False)
      .agg(
          authors=(author_for_posts, lambda s: s.astype(str).fillna("").nunique()),
          posts=(author_for_posts, "size")
      )
      .reset_index()
      .rename(columns={"_brand": "Primary_Brand"})
)

brand_stats["%_authors"] = (brand_stats["authors"] / total_authors_global * 100).round(2)
brand_stats["%_posts"]   = (brand_stats["posts"]   / total_posts_global   * 100).round(2)
brand_stats["authors_per_post"] = (brand_stats["authors"] / brand_stats["posts"]).round(3)

brand_stats = brand_stats.sort_values(["authors", "posts"], ascending=[False, False]).reset_index(drop=True)

print("\n=== Per-brand: authors vs posts ===")
print(brand_stats)

brand_stats.to_csv("Results/brand_authors_vs_posts.csv", index=False)

#_-------------------------

# ---------- per-brand + dose: authors vs posts ----------
brand_dose_stats = (
    df.assign(_brand=df[brand_col].astype(str).fillna(""), _dose=df[bucket_col].astype(str).fillna(""))
      .groupby(["_brand", "_dose"], dropna=False)
      .agg(
          authors=(author_col, lambda s: s.astype(str).nunique()),
          posts=(author_col, "size")
      )
      .reset_index()
      .rename(columns={"_brand": "Primary_Brand", "_dose": "Dose_Bucket"})
)

brand_dose_stats["%_authors"] = (brand_dose_stats["authors"] / total_authors_global * 100).round(2)
brand_dose_stats["%_posts"]   = (brand_dose_stats["posts"] / total_posts_global * 100).round(2)
brand_dose_stats["authors_per_post"] = (brand_dose_stats["authors"] / brand_dose_stats["posts"]).round(3)

brand_dose_stats = brand_dose_stats.sort_values(["authors", "posts"], ascending=[False, False]).reset_index(drop=True)

brand_dose_stats.to_csv("Results/brand_dose_authors_vs_posts.csv", index=False)
print("\n=== Per-brand + dose: authors vs posts ===")
print(brand_dose_stats.head(20))  # limit for display



#--------- account type analysis
'''def majority_or_tiebreak(series):
    vc = series.value_counts()
    if vc.empty:
        return "individual"          # not expected, but safe default
    if len(vc) == 1 or (len(vc) > 1 and vc.iloc[0] != vc.iloc[1]):
        return vc.idxmax()           # clear majority
    return "individual"              # tie → individual

if acct_col in ann.columns:
    acct_by_author = (
        ann[[author_col, acct_col]].dropna()
           .drop_duplicates()
           .groupby(author_col)[acct_col]
           .agg(majority_or_tiebreak)
           .to_frame("account_type")
    )
    
else:
    acct_by_author = pd.DataFrame(index=pd.Index(ann[author_col].unique(), name=author_col))
    '''
'''if acct_col in ann.columns:
    acct_by_author = (
        ann[[author_col, acct_col]]
        .dropna()
        .drop_duplicates()
        .groupby(author_col, sort=False)[acct_col]
        .first()                # <-- first account type per author
        .to_frame("account_type")
    )
else:
    acct_by_author = pd.DataFrame(index=pd.Index(ann[author_col].unique(), name=author_col))

# join into authors
authors = (pd.DataFrame(index=pd.Index(ann[author_col].unique(), name=author_col)).sort_index()
           .join(brand_by_author,    how="left")
           .join(dose_by_author,     how="left")
           .join(offlabel_by_author, how="left")
           .join(acct_by_author,     how="left"))


# join into authors (add to your existing join chain)
authors = (pd.DataFrame(index=pd.Index(ann[author_col].unique(), name=author_col)).sort_index()
           .join(brand_by_author,    how="left")
           .join(dose_by_author,     how="left")
           .join(offlabel_by_author, how="left")
           .join(acct_by_author,     how="left"))     # <-- added
# keep  fills for any_brand/any_dose/any_offlabel
authors["any_brand"]    = authors["any_brand"].fillna(False).astype(bool)
authors["any_dose"]     = authors["any_dose"].fillna(False).astype(bool)
authors["any_offlabel"] = authors["any_offlabel"].fillna(False).astype(bool)

# account-type breakdown among authors who mentioned ≥1 brand
brand_authors = authors.loc[authors["any_brand"]]

n_individual     = int((brand_authors["account_type"] == "individual").sum())
n_organisational = int((brand_authors["account_type"] == "organisational").sum())
total_with_brand = int(len(brand_authors))

pct2 = lambda x: round(100.0 * x / total_with_brand, 2) if total_with_brand else 0.0

print("\n=== Account type among authors who mentioned ≥1 brand ===")
print(f"individual:      {n_individual} ({pct2(n_individual)}%)")
print(f"organisational:  {n_organisational} ({pct2(n_organisational)}%)")

# restrict to authors with off-label mentions
offlabel_authors = authors.loc[authors["any_offlabel"]]

n_individual     = int((offlabel_authors["account_type"] == "individual").sum())
n_organisational = int((offlabel_authors["account_type"] == "organisational").sum())
total_offlabel   = int(len(offlabel_authors))

pct = lambda x: round(100.0 * x / total_offlabel, 2) if total_offlabel else 0.0

print("\n=== Account type among authors who mentioned any off-label dose ===")
print(f"individual:      {n_individual} ({pct(n_individual)}%)")
print(f"organisational:  {n_organisational} ({pct(n_organisational)}%)")


# restrict to authors who mentioned both a brand and any dose
brand_dose_authors = authors.loc[authors["any_brand"] & authors["any_dose"]]

n_individual     = int((brand_dose_authors["account_type"] == "individual").sum())
n_organisational = int((brand_dose_authors["account_type"] == "organisational").sum())
total_both       = int(len(brand_dose_authors))

pct = lambda x: round(100.0 * x / total_both, 2) if total_both else 0.0

print("\n=== Account type among authors mentioning both brand & (any) dose ===")
print(f"individual:      {n_individual} ({pct(n_individual)}%)")
print(f"organisational:  {n_organisational} ({pct(n_organisational)}%)")'''
