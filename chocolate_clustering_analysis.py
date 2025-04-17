
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# Load the data
FILEPATH = 'flavors_of_cacao_dataframe.xlsx'
df = pd.read_excel(FILEPATH)

# Clean up column names
df.columns = [col.strip().replace('\n', ' ').replace('\r', '').replace('\u00a0', ' ').strip() for col in df.columns]
print("Cleaned column names:")
print(df.columns)

# Convert 'Cocoa Percent' to numeric if needed
if df['Cocoa Percent'].dtype == 'object':
    df['Cocoa Percent'] = df['Cocoa Percent'].str.replace('%', '').astype(float) / 100

# Select only numeric columns for clustering
numeric_cols = ['REF', 'Review Date', 'Cocoa Percent', 'Rating']
df_numeric = df[numeric_cols]

print("Numeric data for clustering:")
print(df_numeric.head())

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

print("Standardized data (first 5 rows):")
print(df_scaled[:5])

# Use elbow method to determine the optimal number of clusters
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method For Optimal k', fontsize=14)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('elbow_curve.png')
plt.close()

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
k_range = range(2, 11)  # Silhouette score requires at least 2 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score For Optimal k', fontsize=14)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('silhouette_scores.png')
plt.close()

# Perform k-means clustering with k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Display the count of samples in each cluster
print("Number of samples in each cluster:")
print(df['Cluster'].value_counts())

# Create a PCA model to reduce dimensions for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

# Plot the clusters in the PCA space
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=60, alpha=0.8)
plt.title('Clusters Visualized using PCA', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('pca_clusters.png')
plt.close()

# Calculate the mean values of features for each cluster
cluster_means = df.groupby('Cluster')[numeric_cols].mean()
print("Mean values of features for each cluster:")
print(cluster_means)

# Visualize the cluster means
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_means, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title('Mean Feature Values by Cluster', fontsize=16)
plt.savefig('cluster_means_heatmap.png')
plt.close()

# Create boxplots to compare the distribution of features across clusters
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Distributions by Cluster', fontsize=16)

for i, feature in enumerate(numeric_cols):
    row, col = i // 2, i % 2
    sns.boxplot(x='Cluster', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Cluster', fontsize=14)
    axes[row, col].set_xlabel('Cluster', fontsize=12)
    axes[row, col].set_ylabel(feature, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('feature_boxplots.png')
plt.close()

# Count of company locations by cluster
location_cluster = pd.crosstab(df['Company Location'], df['Cluster'])
location_cluster_pct = location_cluster.div(location_cluster.sum(axis=1), axis=0) * 100

# Display the top 10 company locations by total count
top_locations = df['Company Location'].value_counts().head(10).index
print("Top company locations by cluster:")
print(location_cluster.loc[top_locations].sort_values(by=0, ascending=False))

# Count of bean origins by cluster
origin_cluster = pd.crosstab(df['Broad Bean Origin'], df['Cluster'])
origin_cluster_pct = origin_cluster.div(origin_cluster.sum(axis=1), axis=0) * 100

# Display the top 10 bean origins by total count
top_origins = df['Broad Bean Origin'].value_counts().head(10).index
print("Top bean origins by cluster:")
print(origin_cluster.loc[top_origins].sort_values(by=0, ascending=False))

print("Analysis complete. All visualizations have been saved as PNG files.")
