import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

csv_file_path = 'edg.csv'
data = pd.read_csv(csv_file_path)

if 'Source' not in data.columns or 'Target' not in data.columns:
    raise ValueError("The CSV file must contain 'Source' and 'Target' columns.")


G = nx.Graph()
edges = list(zip(data['Source'], data['Target']))
G.add_edges_from(edges)


communities = nx.algorithms.community.label_propagation_communities(G)


communities = list(communities)


sorted_communities = sorted(communities, key=len, reverse=True)[:5]

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
plt.title("Graph with Top 5 Label Propagation Communities")
plt.show()

print(f"Number of communities detected: {len(sorted_communities)}")
for i, community in enumerate(sorted_communities):
    print(f"Community {i+1}: {list(community)}")
