import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv('Data/mallCustomerData.txt', sep=',')
print(data.head())

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], c='black', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Data')
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters
colors = ['r', 'g', 'b', 'y', 'c']
for i in range(5):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Customers')
plt.legend()
plt.show()
