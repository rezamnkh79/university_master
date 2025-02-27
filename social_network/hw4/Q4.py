import pandas as pd
import networkx as nx
from cdlib import algorithms
import matplotlib.pyplot as plt

csv_file_path = 'edg.csv'
data = pd.read_csv(csv_file_path)

G = nx.Graph()
edges = list(zip(data['Source'], data['Target']))
G.add_edges_from(edges)

infomap_communities = algorithms.infomap(G)

sorted_communities = sorted(infomap_communities.communities, key=len, reverse=True)[:4]

nodes_to_draw = set(node for community in sorted_communities for node in community)
subgraph = G.subgraph(nodes_to_draw)

colors = plt.cm.get_cmap('tab10', len(sorted_communities)).colors
node_colors = {}
for i, community in enumerate(sorted_communities):
    for node in community:
        node_colors[node] = colors[i % len(colors)]

pos = nx.spring_layout(subgraph)
nx.draw(
    subgraph, pos,
    node_color=[node_colors[node] for node in subgraph.nodes()],
    with_labels=False, node_size=50, edge_color="gray"
)
plt.title("Graph with Top 4 Infomap Communities")
plt.show()
