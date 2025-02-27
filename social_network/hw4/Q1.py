import pandas as pd
import networkx as nx

df = pd.read_csv('edg.csv')

G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr='Weight')

k_shell = nx.core_number(G)


df['K-Shell'] = df['Source'].map(k_shell)

communities_by_k_shell = {}
for node, k in k_shell.items():
    if k not in communities_by_k_shell:
        communities_by_k_shell[k] = []
    communities_by_k_shell[k].append(node)

df['Community'] = df['Source'].map(lambda x: k_shell[x])

df.to_csv('k_shell_result.csv', index=False)
print(df.head())
