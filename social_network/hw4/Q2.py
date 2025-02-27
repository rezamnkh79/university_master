import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


csv_file_path = 'edg.csv'
data = pd.read_csv(csv_file_path)

G = nx.from_pandas_edgelist(data, source='Source', target='Target')

L = nx.laplacian_matrix(G).astype(float)
k = 2
vals, vecs = eigs(L, k=k, which='SM')


embedding = np.real(vecs)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(embedding)

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'yellow', 'purple']
pos = nx.spring_layout(G)

for i in range(k):
    cluster_nodes = [n for n, label in zip(G.nodes(), labels) if label == i]
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_color=colors[i], label=f"Cluster {i+1}")

nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.legend()
plt.title("Spectral Partitioning Clusters")
plt.show()
