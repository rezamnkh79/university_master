import time
from math import log

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# def create_powerlaw_graph(size, gamma):
#     G = nx.barabasi_albert_graph(N, m=3)
#     degree_sequence = [d for _, d in G.degree()]
#     return G
def create_powerlaw_graph(N, gamma):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    degree_sequence = np.zeros(N)
    for node in range(1, N):
        probabilities = degree_sequence[:node] ** (-gamma)
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        if np.sum(probabilities) == 0:
            probabilities = np.ones(node) / node
        probabilities /= probabilities.sum()
        target_node = np.random.choice(range(node), p=probabilities)
        G.add_edge(node, target_node)
        degree_sequence[node] = 1
        degree_sequence[target_node] += 1
    return G


def calculate_distance(graph):
    connected_components = nx.connected_components(graph)
    result = []
    for cc in connected_components:
        cc = graph.subgraph(cc).copy()
        result.append(nx.average_shortest_path_length(cc) * len(cc.nodes))
    return sum(result) / len(graph.nodes)


def calculate_expected_dist(size, gamma):
    assert gamma >= 2
    if gamma <= 2.0001:
        return 2.7
    elif gamma < 3:
        return ((0.8) * log(log(size))) / log(gamma - 1)
    elif gamma == 3:
        return ((1.46) * log(size)) / log(log(size))
    else:
        return (0.715) * log(size)


def calculate_distance_sampling(graph):
    sampling_size = int(len(graph.nodes) + 1) + 1
    nodes = np.array(graph.nodes)
    d = 0
    for i in range(sampling_size):
        selected_node = np.random.choice(nodes, size=2, replace=True)
        while selected_node[0] == selected_node[1]:
            selected_node = np.random.choice(nodes, size=2, replace=True)
        d += nx.shortest_path_length(graph,
                                     source=selected_node[0],
                                     target=selected_node[1])
    return d / sampling_size


def plot(sizes, gamma, result):
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(sizes, result, label=f'actual', color='#4322b1')
    plt.plot(sizes, [calculate_expected_dist(size, gamma) for size in sizes]
             , label='expected', color='#f07aa1')

    plt.title(f'Power-law Distribution for γ={gamma}')
    plt.xlabel('N')
    plt.ylabel('<d>')
    plt.legend()
    plt.show()


def generate_ba_graph(n, m):
    return nx.barabasi_albert_graph(n, m)


def plot_path_length_distribution(graph, gamma):
    path_lengths = dict(nx.shortest_path_length(graph))
    all_lengths = []

    for node, lengths in path_lengths.items():
        all_lengths.extend(lengths.values())

    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=30, alpha=0.75, edgecolor='black')
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.title(f'Path Length Distribution for Gamma (m) = {gamma}')
    plt.grid(True)
    plt.show()


graphs = {}
dists = {}

gammas = [2, 2.5, 3, 3.5]
sizes = [2000, 4000, 6000, 8000, 10000]

for gamma in gammas:
    if gamma not in graphs.keys():
        graphs[gamma] = {}
    if gamma not in dists.keys():
        dists[gamma] = {}
    for size in sizes:
        if size in graphs[gamma].keys():
            continue
        print("---------------------------------------------------------------")
        print(f"Generating graph for size:{size}, gamma:{gamma} ...")
        start = time.time()
        g = create_powerlaw_graph(size, gamma)
        end = time.time()
        print(f"Graph generated in {round(end - start, 2)}s.")
        print(f"Cleaning graph ...")
        start = time.time()
        g = nx.Graph(g)
        graphs[gamma][size] = g
        end = time.time()
        print(f"Graph cleaned in {round(end - start, 2)}s.")
        print(f"Calculating graph average distance ...")
        start = time.time()
        dists[gamma][size] = nx.average_shortest_path_length(g)
        end = time.time()
        print(
            f"calculated Graph average distance with value of {round(dists[gamma][size], 2)} in {round(end - start, 2)}s.")

plot(sizes, gammas[0], list(dists[gammas[0]].values()))
plot(sizes, gammas[1], list(dists[gammas[1]].values()))
plot(sizes, gammas[2], list(dists[gammas[2]].values()))
plot(sizes, gammas[3], list(dists[gammas[3]].values()))

gammas = [2.0001, 2.5, 3, 3.5]
sizes_s = [i * 500 for i in range(1, 21)]
graphs_s = {}
dists_s = {}
for gamma in gammas:
    if gamma not in graphs_s.keys():
        graphs_s[gamma] = {}
    if gamma not in dists_s.keys():
        dists_s[gamma] = {}
    for size in sizes_s:
        print("---------------------------------------------------------------")
        print(f"Generating graph for size:{size}, gamma:{gamma} ...")
        start = time.time()
        g = create_powerlaw_graph(size, gamma)
        end = time.time()
        print(f"Graph generated in {round(end - start, 2)}s.")
        print(f"Cleaning graph ...")
        start = time.time()
        g = nx.Graph(g)
        graphs_s[gamma][size] = g
        end = time.time()
        print(f"Graph cleaned in {round(end - start, 2)}s.")
        print(f"Calculating graph average distance ...")
        start = time.time()
        dists_s[gamma][size] = calculate_distance_sampling(g)
        end = time.time()
        print(
            f"calculated Graph average distance with value of {round(dists_s[gamma][size], 2)} in {round(end - start, 2)}s.")

plot(sizes_s, gammas[0], list(dists_s[gammas[0]].values()))
plot(sizes_s, gammas[1], list(dists_s[gammas[1]].values()))
plot(sizes_s, gammas[2], list(dists_s[gammas[2]].values()))
plot(sizes_s, gammas[3], list(dists_s[gammas[3]].values()))

n = 10000  # Number of nodes
gamma_values = [2.0001, 2.5, 3, 3.5]

for gamma in gamma_values:
    m = int(gamma)
    graph = generate_ba_graph(n, m)
    plot_path_length_distribution(graph, gamma)
    avg_path_length = nx.average_shortest_path_length(graph)
    print(f'Gamma (m): {gamma}, Average Shortest Path Length: {avg_path_length}')
