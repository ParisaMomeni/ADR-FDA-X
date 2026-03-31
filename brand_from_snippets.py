import pandas as pd
import re

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

def extract_brand(text):
    text = str(text).lower()
    text_clean = re.sub(r'[\.\,\!\?\-\_\(\)]', ' ', text)
    for brand, aliases in brand_aliases.items():
        for alias in aliases:
            if re.search(r'\b{}\b'.format(re.escape(alias)), text_clean):
                return brand
    return "Unknown"

def summarize_brands(snippets):
    df = pd.DataFrame({'Snippet': snippets})
    df["Brand"] = df["Snippet"].apply(extract_brand)
    summary = df.groupby("Brand").size().reset_index(name="Count")
    total = summary["Count"].sum()
    summary["Percentage"] = (summary["Count"] / total * 100).round(2)
    summary = summary.sort_values("Count", ascending=False)
    return summary, df

if __name__ == "__main__":
    df = pd.read_pickle("data/1adrs2.pkl")
    snippets = df["Snippet"].tolist()
    summary, labeled_df = summarize_brands(snippets)
    print(summary)
    summary.to_csv("Results/brand_from_snippet.csv", index=False)
    labeled_df.to_csv("Results/posts_with_brand.csv", index=False)
