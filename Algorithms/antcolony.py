import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# Node mapping
node_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
node_indices = {name: idx for idx, name in enumerate(node_names)}

# Distance matrix (0 means no direct connection)
distances = np.array([
    [0, 15, 10, 17,  0,  0,  7,  0],  # A
    [15, 0,  0, 12,  0,  0,  0,  0],  # B
    [10, 0,  0,  5,  0,  0,  7,  0],  # C
    [17,12,  5,  0,  2, 10,  0,  4],  # D
    [0,  0,  0,  2,  0,  0,  0,  0],  # E
    [0,  0,  0, 10,  0,  0,  0, 11],  # F
    [7,  0,  7,  0,  0,  0,  0, 25],  # G
    [0,  0,  0,  4,  0, 11, 25,  0],  # H
], dtype=float)

# Replace 0s (except diagonal) with np.inf for easier calculations
for i in range(len(distances)):
    for j in range(len(distances)):
        if i != j and distances[i][j] == 0:
            distances[i][j] = np.inf

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run_and_plot_ants(self, start_idx, goal_idx, node_names, plot_each_ant=True):
        all_time_shortest_path = (None, np.inf)
        for iteration in range(self.n_iterations):
            all_paths = []
            for ant_num in range(self.n_ants):
                path, cost = self.gen_path(start_idx, goal_idx)
                if path is not None:
                    all_paths.append((path, cost))
                    if plot_each_ant:
                        plot_ant_path(path, ant_num, iteration, node_names, self.distances)
            if not all_paths:
                continue
            self.spread_pheromone(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= (1 - self.decay)
        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, cost in sorted_paths[:n_best]:
            for move in zip(path[:-1], path[1:]):
                self.pheromone[move] += 1.0 / cost

    def gen_path(self, start, end):
        path = [start]
        visited = set(path)
        total_cost = 0
        while path[-1] != end:
            move = self.pick_move(self.pheromone[path[-1]], self.distances[path[-1]], visited)
            if move is None:
                return None, np.inf  # Dead end
            path.append(move)
            visited.add(move)
            total_cost += self.distances[path[-2]][move]
            if len(path) > len(self.distances)*2:  # Prevent infinite loops
                return None, np.inf
        return path, total_cost

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        dist = np.copy(dist)
        dist[list(visited)] = np.inf
        row = (pheromone ** self.alpha) * ((1.0 / dist) ** self.beta)
        if row.sum() == 0:
            return None
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

def plot_ant_path(path, ant_num, iteration, node_names, distances):
    G = nx.Graph()
    for i in range(len(node_names)):
        for j in range(i+1, len(node_names)):
            if distances[i][j] != np.inf:
                G.add_edge(node_names[i], node_names[j], weight=distances[i][j])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if path is not None:
        path_names = [node_names[i] for i in path]
        path_edges = list(zip(path_names, path_names[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=3)
    plt.title(f"Iteration {iteration+1}, Ant {ant_num+1}: {' -> '.join([node_names[i] for i in path])}")
    plt.show()

# --- Main Program ---
print("Nodes:", node_names)
start = input("Enter start node: ").strip().upper()
goal = input("Enter goal node: ").strip().upper()
if start not in node_names or goal not in node_names:
    print("Invalid node name(s).")
    exit()
start_idx = node_indices[start]
goal_idx = node_indices[goal]

print("\nChoose scenario:")
print("1. No.of.Ants = 3, alpha = 0.5, beta = 3")
print("2. No.of.Ants = 5, alpha = 0.5, beta = 7")
scenario = input("Enter 1 or 2: ").strip()
if scenario == '1':
    n_ants = 3
    alpha = 0.5
    beta = 3
elif scenario == '2':
    n_ants = 5
    alpha = 0.5
    beta = 7
else:
    print("Invalid scenario choice.")
    exit()

# Use n_iterations=2 for demonstration (change to higher for full run, but beware of many plots)
aco = AntColony(distances, n_ants=n_ants, n_best=max(1, n_ants//2), n_iterations=2, decay=0.1, alpha=alpha, beta=beta)
best_path, best_cost = aco.run_and_plot_ants(start_idx, goal_idx, node_names, plot_each_ant=True)

if best_cost == np.inf:
    print(f"No path found from {start} to {goal}.")
else:
    path_names = [node_names[i] for i in best_path]
    print(f"\nBest path from {start} to {goal}: {' -> '.join(path_names)}")
    print(f"Total transmission cost: {best_cost:.2f}")
