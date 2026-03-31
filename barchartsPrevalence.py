import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'  # Use Times-like font
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = 1.0

df_test = pd.read_pickle("data/processed_data.pkl")
before_retweets = len(df_test)

df_test = df_test[df_test['Engagement Type'] != 'RETWEET'] #
after_retweets = len(df_test)
# Print results
print(f"Rows before removing retweets: {before_retweets}")
print(f"Rows after removing retweets: {after_retweets}")
print(f"Number of retweets removed: {before_retweets - after_retweets}")

#df = pd.read_pickle("data/1adrs2.pkl")
df = pd.read_excel("data/processed_data_merged_noRT_adrs_threshold0.55.xlsx")
import ast

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
'''def adr_str_to_list(adr_str):
    if adr_str == "[]":
        return []
    adr_str = adr_str.replace('[','')
    adr_str = adr_str.replace(']','')
    adr_str = adr_str.replace("'",'')
    return adr_str.split(',')'''

df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(adr_str_to_list)
#df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(adr_str_to_list)


# Step 1: Count ADR mentions (unique per post)
adr_post_counts = defaultdict(int)
for adrs in df["Extracted_ADRs"]:
    for adr in set(adrs):  # avoid double-counting ADRs in a single post
        adr_post_counts[adr] += 1

# Step 2: Create prevalence DataFrame
total_posts = len(df)
prevalence_df = pd.DataFrame([
    {"ADR": adr, 
     "Mentions": count, 
     "Prevalence (%)": round(100 * count / total_posts, 2)}
    for adr, count in adr_post_counts.items()
])
prevalence_df = prevalence_df.sort_values(by="Prevalence (%)", ascending=False)

# Step 3: Save table
prevalence_df.to_csv("Results/adr_prevalence_table.csv", index=False)

# Step 4: Plot Top 10 ADRs by Prevalence
top_n = 11
top_prevalent = prevalence_df.head(top_n)

plt.figure(figsize=(8, 5))
#plt.bar(top_prevalent["ADR"], top_prevalent["Prevalence (%)"], color='skyblue')
plt.bar(top_prevalent["ADR"], top_prevalent["Prevalence (%)"], color='#00a19a', edgecolor='black')

#plt.title(f"Top {top_n} ADRs by Prevalence")
plt.ylabel("Prevalence (%)", fontsize = 15.5, fontweight='bold')
#plt.xlabel("ADR", fontsize=14)
#plt.xticks(rotation=45)
#plt.xticks(rotation=45, fontsize=18)
plt.xticks(ticks=range(len(top_prevalent)), labels=[adr.title() for adr in top_prevalent["ADR"]],
           rotation=45, ha='right', fontsize=15.5, fontweight='bold')

plt.yticks(fontsize=15.5, fontweight='bold')

plt.ylim(0, top_prevalent["Prevalence (%)"].max() * 1.15)  # add headroom

plt.grid(axis='y', linestyle='--', alpha=0.4)

# Optional annotation of values
#for idx, val in enumerate(top_prevalent["Prevalence (%)"]):
  #  plt.text(idx, val + 0.2, f"{val:.1f}%", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("adr_prevalence_barplot.png")
plt.show()
