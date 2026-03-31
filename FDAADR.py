import pandas as pd
import matplotlib.pyplot as plt

def clean_adr_data(df, col1, col2=None):
    # Keep only numeric rows
    if col2:
        df = df[
            df[col1].apply(lambda x: isinstance(x, (float, int))) &
            df[col2].apply(lambda x: isinstance(x, (float, int)))
        ]
    else:
        df = df[df[col1].apply(lambda x: isinstance(x, (float, int)))]
    return df.reset_index(drop=True)

# ✅ Replace with your actual file path on local machine
file_path = "data/FDA_ADR_Prevalences.xlsx"

sheet_name = "Ozempic, Wegovy, and Rybelsus"
df_all = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)
print("Actual columns:", df_all.columns.tolist())

# Define subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
bar_width = 0.35

# --------------------------------------------
# Ozempic Plot
# --------------------------------------------
ozempic = df_all[["ADR", "FDA 0.5 mg", "FDA 1 mg"]].dropna()
ozempic.columns = ["ADR", "0.5 mg", "1 mg"]
ozempic = clean_adr_data(ozempic, "0.5 mg", "1 mg")
x = range(len(ozempic))

axes[0].bar([i - bar_width/2 for i in x], ozempic["0.5 mg"]*100, width=bar_width, label="0.5 mg")
axes[0].bar([i + bar_width/2 for i in x], ozempic["1 mg"]*100, width=bar_width, label="1 mg")
axes[0].set_xticks(x)
axes[0].set_xticklabels(ozempic["ADR"], rotation=45, ha="right")
axes[0].set_title("Ozempic")
axes[0].legend()
axes[0].set_ylabel("Prevalence (%)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# --------------------------------------------
# Wegovy Plot
# --------------------------------------------
wegovy = df_all[["ADR", "FDA 2.4 mg"]].dropna()
wegovy.columns = ["ADR", "2.4 mg"]
wegovy = clean_adr_data(wegovy, "2.4 mg")
x = range(len(wegovy))

axes[1].bar(x, wegovy["2.4 mg"]*100, width=0.6, label="2.4 mg", color="#00a19a")
axes[1].set_xticks(x)
axes[1].set_xticklabels(wegovy["ADR"], rotation=45, ha="right")
axes[1].set_title("Wegovy")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# --------------------------------------------
# Rybelsus Plot
# --------------------------------------------
rybelsus = df_all[["ADR", "FDA 7 mg", "FDA 14 mg"]].dropna()
rybelsus.columns = ["ADR", "7 mg", "14 mg"]
rybelsus = clean_adr_data(rybelsus, "7 mg", "14 mg")
x = range(len(rybelsus))

axes[2].bar([i - bar_width/2 for i in x], rybelsus["7 mg"]*100, width=bar_width, label="7 mg")
axes[2].bar([i + bar_width/2 for i in x], rybelsus["14 mg"]*100, width=bar_width, label="14 mg")
axes[2].set_xticks(x)
axes[2].set_xticklabels(rybelsus["ADR"], rotation=45, ha="right")
axes[2].set_title("Rybelsus")
axes[2].legend()
axes[2].grid(axis="y", linestyle="--", alpha=0.5)

# --------------------------------------------
# Final touches
# --------------------------------------------
fig.suptitle("ADR Prevalence Comparison Across Brands", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("ADR_FDA_Bar_Charts.png", dpi=300)
plt.show()
