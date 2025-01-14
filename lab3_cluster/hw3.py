import numpy as np
# import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the word vectors
data_file = "nmt-en-dec-512.txt"
data = []
words = []

with open(data_file, "r") as f:
    for line in f:
        parts = line.strip().split('\t')
        words.append(parts[0])
        data.append(list(map(float, parts[1:])))

word_vectors = np.array(data)

print("Number of words:", len(words))
print("Dimension of word vectors:", word_vectors.shape[1])


# Task 1: T-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(word_vectors)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], s=10, color="grey")
plt.title("T-SNE Visualization of Word Vectors")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
# plt.savefig("tsne_visualization.png")
# plt.show()


# Task 2: Clustering and Evaluation
n_clusters = [3, 5, 10]
linkage_methods = ['ward', 'single', 'complete']

for n in n_clusters:
    # K-means Clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans_labels = kmeans.fit_predict(word_vectors)
    silhouette_kmeans = silhouette_score(word_vectors, kmeans_labels)

    # Gaussian Mixture Clustering
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm_labels = gmm.fit_predict(word_vectors)
    silhouette_gmm = silhouette_score(word_vectors, gmm_labels)

    print(f"t-SNE -> K-means (n_clusters={n}): Silhouette Coefficient = {silhouette_kmeans:.3f}")
    print(f"t-SNE -> Gaussian Mixture (n_clusters={n}): Silhouette Coefficient = {silhouette_gmm:.3f}")

    for linkage in linkage_methods:
        agglo = AgglomerativeClustering(n_clusters=n, linkage=linkage)
        agglo_labels = agglo.fit_predict(word_vectors)
        silhouette_agglo = silhouette_score(word_vectors, agglo_labels)
        print(f"t-SNE -> Agglomerative (n_clusters={n}, linkage={linkage}): Silhouette Coefficient = {silhouette_agglo:.3f}")

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)

# Plot K-Means Clusters
sns.scatterplot(ax=axes[0], x=tsne_results[:, 0], y=tsne_results[:, 1], hue=kmeans_labels+1, palette='Set2', s=10)
axes[0].set_title("T-SNE with K-Means Clusters")
axes[0].set_xlabel("TSNE Component 1")
axes[0].set_ylabel("TSNE Component 2")
axes[0].legend(title="Clusters", loc='best', fontsize='small')

# Plot Gaussian Mixture Clusters
sns.scatterplot(ax=axes[1], x=tsne_results[:, 0], y=tsne_results[:, 1], hue=gmm_labels+1, palette='Set2', s=10)
axes[1].set_title("T-SNE with Gaussian Mixture Clusters")
axes[1].set_xlabel("TSNE Component 1")
axes[1].set_ylabel("TSNE Component 2")
axes[1].legend(title="Clusters", loc='best', fontsize='small')

# Plot Agglomerative Clustering Clusters
sns.scatterplot(ax=axes[2], x=tsne_results[:, 0], y=tsne_results[:, 1], hue=agglo_labels+1, palette='Set2', s=10)
axes[2].set_title("T-SNE with Agglomerative Clustering Clusters Complete linkage")
axes[2].set_xlabel("TSNE Component 1")
axes[2].set_ylabel("TSNE Component 2")
axes[2].legend(title="Clusters", loc='best', fontsize='small')

# Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig("tsne_cluster_visualization.png")
# plt.show()


# Task 3: PCA Transformation and Visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(word_vectors)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], s=10, color="grey")
plt.title("PCA Visualization of Word Vectors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
# plt.savefig("pca_visualization.png")
# plt.show()

# K-means and GMM on PCA-transformed data
for n in n_clusters:
    kmeans_pca = KMeans(n_clusters=n, random_state=42)
    kmeans_pca_labels = kmeans_pca.fit_predict(pca_results)
    silhouette_kmeans_pca = silhouette_score(pca_results, kmeans_pca_labels)

    gmm_pca = GaussianMixture(n_components=n, random_state=42)
    gmm_pca_labels = gmm_pca.fit_predict(pca_results)
    silhouette_gmm_pca = silhouette_score(pca_results, gmm_pca_labels)

    print(f"PCA -> K-means (n_clusters={n}): Silhouette Coefficient = {silhouette_kmeans_pca:.3f}")
    print(f"PCA -> Gaussian Mixture (n_clusters={n}): Silhouette Coefficient = {silhouette_gmm_pca:.3f}")

# Create a figure with three subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 8), dpi=300)

# Plot K-Means Clusters
sns.scatterplot(ax=axes[0], x=pca_results[:, 0], y=pca_results[:, 1], hue=kmeans_pca_labels+1, palette='Set2', s=10)
axes[0].set_title("PCA with K-Means Clusters")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].legend(title="Clusters", loc='best', fontsize='small')

# Plot Gaussian Mixture Clusters
sns.scatterplot(ax=axes[1], x=pca_results[:, 0], y=pca_results[:, 1], hue=gmm_pca_labels+1, palette='Set2', s=10)
axes[1].set_title("PCA with Gaussian Mixture Clusters")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].legend(title="Clusters", loc='best', fontsize='small')

# Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig("pca_cluster_visualization.png")
# plt.show()



# Task 4: Visualize PCA Component Distributions
n_components = 5
pca_full = PCA(n_components=n_components)
pca_full_results = pca_full.fit_transform(word_vectors)


# Loop through the first 5 PCA components
for i in range(n_components):
    # Divide PCA values into 100 bins
    bins = np.linspace(pca_full_results[:, i].min(), pca_full_results[:, i].max(), 101)
    bin_indices = np.digitize(pca_full_results[:, i], bins) - 1

    # print(bin_indices)
    # print(bins)

    # Count word frequencies per bin and sample representative words
    word_bins = {j: [] for j in range(101)}
    for idx, bin_idx in enumerate(bin_indices.tolist()):
        word_bins[bin_idx].append(words[idx])

    # Prepare data for plotting
    bin_counts = [len(word_bins[j]) for j in range(101)]
    bin_labels = [" | ".join(word_bins[j][:3]) for j in range(101)]  # Up to 3 sample words per bin

    # Plot histogram as a bar chart
    plt.figure(figsize=(20, 6))
    plt.bar(range(101), bin_counts, alpha=0.75, color="grey")
    plt.xticks(range(101), bin_labels, rotation=90, fontsize=8)
    plt.title(f"Distribution of Words along PCA Component {i + 1}")
    plt.xlabel(f"PCA Component {i + 1} (Sampled Words per Bin)")
    plt.ylabel("Frequency")
    # plt.tight_layout()
    # plt.savefig(f"pca_c{i+1}_distribution.png")
    # plt.show()



# Task 5: ICA Transformation and Visualization
ica = FastICA(n_components=50, random_state=42)
ica_results = ica.fit_transform(word_vectors)

# for i in range(5):
#     plt.figure(figsize=(10, 6))
#     plt.hist(ica_results[:, i], bins=101, alpha=0.75, color="grey")
#     plt.title(f"Distribution of Words along ICA Component {i + 1}")
#     plt.xlabel(f"ICA Component {i + 1}")
#     plt.ylabel("Frequency")
#     plt.show()

# Loop through the first 5 ICA components
for i in range(5):
    # Divide ICA values into 100 bins
    bins = np.linspace(ica_results[:, i].min(), ica_results[:, i].max(), 101)
    bin_indices = np.digitize(ica_results[:, i], bins) - 1

    # Count word frequencies per bin and sample representative words
    word_bins = {j: [] for j in range(101)}
    for idx, bin_idx in enumerate(bin_indices.tolist()):
        word_bins[bin_idx].append(words[idx])

    # Prepare data for plotting
    bin_counts = [len(word_bins[j]) for j in range(101)]
    bin_labels = [" | ".join(word_bins[j][:3]) for j in range(101)]  # Up to 3 sample words per bin

    # Plot histogram as a bar chart
    plt.figure(figsize=(20, 6))
    plt.bar(range(101), bin_counts, alpha=0.75, color="grey")
    plt.xticks(range(101), bin_labels, rotation=90, fontsize=8)
    plt.title(f"Distribution of Words along ICA Component {i + 1}")
    plt.xlabel(f"ICA Component {i + 1} (Sampled Words per Bin)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # plt.savefig(f"ica_c{i+1}_distribution.png")
    # plt.show()

