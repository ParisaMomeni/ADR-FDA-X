#1. Sentence Embedding & Semantic Similarity (using sentence-transformers)
#2. Clustering with KMeans
#3. ADR Co-occurrence Network


#pip install sentence-transformers scikit-learn networkx matplotlib seaborn


#1. Sentence Embedding & Semantic Similarity (using sentence-transformers)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Flatten ADRs to unique terms
df = pd.read_pickle("data/1adrs2.pkl")

adr_phrases = df["Extracted_ADRs"].explode().dropna().unique().tolist()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
adr_embeddings = model.encode(adr_phrases, show_progress_bar=True)

# Similarity matrix for top 20 ADRs (for clarity)
similarity_matrix = cosine_similarity(adr_embeddings[:20])
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, xticklabels=adr_phrases[:20], yticklabels=adr_phrases[:20],
            cmap="coolwarm", square=True)
plt.title("Semantic Similarity Heatmap (Top 20 ADRs)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------
#2. Clustering with KMeans


# Load processed ADRs
df = pd.read_pickle("data/processed_data_30k_with_adrs.pkl")

# Get unique ADR terms
adr_phrases = df["Extracted_ADRs"].explode().dropna().unique().tolist()
print(f"Number of unique ADRs: {len(adr_phrases)}")

# Embed ADRs
model = SentenceTransformer("all-MiniLM-L6-v2")
adr_embeddings = model.encode(adr_phrases, show_progress_bar=True)

# Cluster using KMeans
n_clusters = min(6, len(adr_embeddings))  # Adjust for small sample
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(adr_embeddings)

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, perplexity=2, random_state=42)  # perplexity < n_samples
reduced_embeddings = tsne.fit_transform(adr_embeddings)

# Plot
plot_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
plot_df["ADR"] = adr_phrases
plot_df["Cluster"] = labels
plt.figure(figsize=(10, 8))
sns.scatterplot(x=plot_df["x"], y=plot_df["y"],
                hue=plot_df["Cluster"], palette="tab10", s=100)
# Annotate each point with the ADR label
for i, row in plot_df.iterrows():
    plt.text(row["x"] + 0.5, row["y"] + 0.5, row["ADR"], fontsize=9)

plt.title("t-SNE Visualization of ADR Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()
for i in range(n_clusters):
    print(f"\n🧪 Cluster {i} contains:")
    cluster_terms = plot_df[plot_df["Cluster"] == i]["ADR"].tolist()
    print(", ".join(cluster_terms))

#---------------------------------------------------------------------------------
#3. ADR Co-occurrence Network
import networkx as nx
import itertools
from collections import Counter

# Co-occurrence matrix
co_occurrence_counts = Counter()
for adrs in df["Extracted_ADRs"]:
    if isinstance(adrs, list) and len(adrs) > 1:
        for pair in itertools.combinations(set(adrs), 2):
            co_occurrence_counts[pair] += 1

# Network graph
G = nx.Graph()
for (adr1, adr2), weight in co_occurrence_counts.items():
    if weight >= 10:
        G.add_edge(adr1, adr2, weight=weight)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
weights = [d['weight'] for _, _, d in G.edges(data=True)]
nx.draw(G, pos, with_labels=True, node_size=500, width=weights, font_size=10)
plt.title("ADR Co-occurrence Network (≥10 co-occurrences)")
plt.show()

# -------------------------------------------------------------------------
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your ADR data
df = pd.read_pickle("data/processed_data_30k_with_adrs.pkl")

# Flatten ADR list to unique terms
adr_phrases = df["Extracted_ADRs"].explode().dropna().unique().tolist()

# Load a SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
adr_embeddings = model.encode(adr_phrases, show_progress_bar=True)

# Reduce to 2D with PCA (for technical visualization)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(adr_embeddings)

# Create DataFrame
plot_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
plot_df["ADR"] = adr_phrases

# Plot
plt.figure(figsize=(14, 10))
plt.scatter(plot_df["PC1"], plot_df["PC2"], s=80, alpha=0.7, edgecolors='w')
for _, row in plot_df.iterrows():
    plt.text(row["PC1"] + 0.05, row["PC2"] + 0.05, row["ADR"], fontsize=8)
plt.title("ADR Semantic Vector Space (PCA Projection)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
#----------------------------
# Required Libraries


# 1. Load processed ADRs
df = pd.read_pickle("data/processed_data_30k_with_adrs.pkl")

# 2. Get unique ADR terms
adr_phrases = df["Extracted_ADRs"].explode().dropna().unique().tolist()
print(f"Number of unique ADR terms: {len(adr_phrases)}")

# 3. Load Sentence-BERT model and compute embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
adr_embeddings = model.encode(adr_phrases, show_progress_bar=True)

# 4. Dimensionality Reduction using PCA (from 384 → 2D)
pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(adr_embeddings)

# 5. Prepare DataFrame for visualization
plot_df = pd.DataFrame(reduced_embeddings, columns=["PC1", "PC2"])
plot_df["ADR"] = adr_phrases

# 6. Plot: PCA-based 2D vector space of ADR terms
plt.figure(figsize=(16, 10))
sns.scatterplot(x="PC1", y="PC2", data=plot_df, s=70, color="steelblue")

# Annotate each ADR term on the plot
for i, row in plot_df.iterrows():
    plt.text(row["PC1"] + 0.01, row["PC2"] + 0.01, row["ADR"], fontsize=9)

plt.title("ADR Semantic Vector Space (PCA Projection)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
import os
os.makedirs("figures", exist_ok=True)

# ✅ Save the figure
plt.savefig("figures/adr_pca_vector_space.png", dpi=300)
plt.show()

#___    --------------------------------------------------------


# Load  processed ADRs
df = pd.read_pickle("data/processed_data_30k_with_adrs.pkl")
adr_phrases = df["Extracted_ADRs"].explode().dropna().unique().tolist()

# Embed ADRs
model = SentenceTransformer("all-MiniLM-L6-v2")
adr_embeddings = model.encode(adr_phrases, show_progress_bar=True)

# PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(adr_embeddings)

# Create figure
plt.figure(figsize=(14, 10))

# Plot vectors from origin to each point
origin = np.zeros((len(reduced_vectors), 2))
plt.quiver(origin[:, 0], origin[:, 1],
           reduced_vectors[:, 0], reduced_vectors[:, 1],
           angles='xy', scale_units='xy', scale=1, alpha=0.6)

# Add labels
for i, txt in enumerate(adr_phrases):
    plt.text(reduced_vectors[i, 0]+0.01, reduced_vectors[i, 1]+0.01, txt, fontsize=9)

plt.title("ADR Vector Embeddings (PCA Arrows from Origin)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/adr_vector_plot_pca.png", dpi=300)
plt.show()

