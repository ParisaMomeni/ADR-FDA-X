import pandas as pd
from ast import literal_eval
import ast

df = pd.read_excel("data/processed_data_merged_noRT_adrs_threshold0.55.xlsx")

# Parse ADRs properly (in case they are stored as strings)
#df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(lambda x: literal_eval(str(x)) if isinstance(x, str) else x)

# Filter rows where Extracted_ADRs is not null and has at least one ADR
#filtered_df = df[df["Extracted_ADRs"].apply(lambda x: isinstance(x, list) and len(x) > 0)]


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
df["Extracted_ADRs"] = df["Extracted_ADRs"].apply(adr_str_to_list)
filtered_df = df[df["Extracted_ADRs"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Save to new file
filtered_df.to_csv("Results/filtered_adrs.csv", index=False)
