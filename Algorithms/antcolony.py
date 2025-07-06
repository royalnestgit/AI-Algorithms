import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class Network:
    def __init__(self, node_names, distance_matrix):
        self.node_names = node_names
        self.node_indices = {name: idx for idx, name in enumerate(node_names)}
        self.distances = np.array(distance_matrix, dtype=float)
        self._prepare_matrix()

    def _prepare_matrix(self):
        for i in range(len(self.distances)):
            for j in range(len(self.distances)):
                if i != j and self.distances[i][j] == 0:
                    self.distances[i][j] = np.inf


class AntColony:
    def __init__(self, network, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        self.network = network
        self.distances = self.network.distances
        self.pheromone = np.ones(self.distances.shape) / len(self.distances)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.node_range = range(len(self.distances))

    def get_successors(self, node_idx, visited):
        return [(i, self.distances[node_idx][i]) for i in range(len(self.distances))
                if self.distances[node_idx][i] != np.inf and i not in visited]

    def is_goal(self, current_idx, goal_idx):
        return current_idx == goal_idx

    def run(self, start_idx, end_idx):
        all_time_shortest_path = (None, np.inf)
        for _ in range(self.n_iterations):
            all_paths = self._gen_all_paths(start_idx, end_idx)
            if not all_paths:
                continue
            self._spread_pheromone(all_paths)
            shortest = min(all_paths, key=lambda x: x[1])
            if shortest[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest
            self.pheromone *= (1 - self.decay)
        return all_time_shortest_path

    def _spread_pheromone(self, paths):
        sorted_paths = sorted(paths, key=lambda x: x[1])
        for path, cost in sorted_paths[:self.n_best]:
            for move in zip(path[:-1], path[1:]):
                self.pheromone[move] += 1.0 / cost

    def _gen_all_paths(self, start, end):
        all_paths = []
        for _ in range(self.n_ants):
            path, cost = self._gen_path(start, end)
            if path:
                all_paths.append((path, cost))
        return all_paths

    def _gen_path(self, start, end):
        path = [start]
        visited = set(path)
        cost = 0
        while not self.is_goal(path[-1], end):
            move = self._pick_move(path[-1], visited)
            if move is None:
                return None, np.inf
            path.append(move)
            visited.add(move)
            cost += self.distances[path[-2]][move]
            if len(path) > len(self.distances) * 2:
                return None, np.inf
        return path, cost

    def _pick_move(self, current, visited):
        successors = self.get_successors(current, visited)
        if not successors:
            return None

        pheromone_values = np.array([self.pheromone[current][i] for i, _ in successors])
        distances = np.array([cost for _, cost in successors])
        heuristic = 1.0 / distances

        numerators = (pheromone_values ** self.alpha) * (heuristic ** self.beta)
        total = np.sum(numerators)
        if total == 0:
            return None

        probabilities = numerators / total
        candidates = [i for i, _ in successors]
        return np.random.choice(candidates, p=probabilities)


def plot_path(network, best_path):
    G = nx.Graph()
    for i in range(len(network.node_names)):
        for j in range(i + 1, len(network.node_names)):
            if network.distances[i][j] != np.inf:
                G.add_edge(network.node_names[i], network.node_names[j], weight=network.distances[i][j])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=600, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if best_path:
        name_path = [network.node_names[i] for i in best_path]
        edges = list(zip(name_path, name_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=3)

    plt.title("ACO - Best Path Visualization")
    plt.show()



if __name__ == "__main__":
    node_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    distance_matrix = [
        [0, 15, 10, 17,  0,  0,  5,  0],
        [15, 0,  0, 12,  0,  0,  0,  0],
        [10, 0,  0,  0,  0,  0,  7,  0],
        [17,12,  5,  0,  2, 10,  0,  4],
        [0,  0,  0,  2,  0,  0,  0,  0],
        [0,  0,  0, 10,  0,  0,  0, 11],
        [7,  0,  7,  0,  0,  0,  0, 25],
        [0,  0,  0,  4,  0, 11, 25,  0]
    ]
    network = Network(node_names, distance_matrix)

    print("Nodes:", node_names)
    start = input("Enter start node: ").strip().upper()
    goal = input("Enter goal node: ").strip().upper()
    if start not in network.node_names or goal not in network.node_names:
        print("Invalid input.")
        exit()

    start_idx = network.node_indices[start]
    goal_idx = network.node_indices[goal]

    print("1. No.of.Ants = 3, alpha = 0.5, beta = 3")
    print("2. No.of.Ants = 5, alpha = 0.5, beta = 7")
    scenario = input("Choose scenario (1/2): ").strip()
    if scenario == '1':
        ants, alpha, beta = 3, 0.5, 3
    elif scenario == '2':
        ants, alpha, beta = 5, 0.5, 7
    else:
        print("Invalid scenario.")
        exit()

    aco = AntColony(network, n_ants=ants, n_best=max(1, ants // 2), n_iterations=100, decay=0.1, alpha=alpha, beta=beta)
    path, cost = aco.run(start_idx, goal_idx)

    if cost == np.inf:
        print(f"No path found from {start} to {goal}.")
    else:
        path_names = [network.node_names[i] for i in path]
        print(f"\nBest path: {' -> '.join(path_names)}")
        print(f"Total cost: {cost:.2f}")
        plot_path(network, path)

        print("\n--- Interpretation ---")
        print(f"ACO found an optimal path from {start} to {goal} minimizing the total transmission cost.")

        n = len(node_names)
        NC = 100
        print("\n--- Complexity ---")
        print(f"Time:  O({NC} × {n}^2 × {ants}) = O({NC * n * n * ants})")
        print(f"Space: O({n}^2 + {n}×{ants}) = O({n * n + n * ants})")
# 