import numpy as np
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

class RBFS:
    def __init__(self, network, heuristic_fn):
        self.network = network
        self.distances = network.distances
        self.heuristic_fn = heuristic_fn

    def search(self, start_idx, goal_idx):
        path = [start_idx]
        f_limit = self.heuristic_fn(start_idx, goal_idx)
        result, final_cost = self._rbfs(start_idx, goal_idx, path, 0, f_limit)
        return result, final_cost

    def _rbfs(self, current, goal_idx, path, g, f_limit):
        print(f"Expanding: {self.network.node_names[current]}, Path: {[self.network.node_names[i] for i in path]}, g: {g}, f_limit: {f_limit}")
        if current == goal_idx:
            return path, g

        successors = []
        for neighbor in self.get_successors(current):
            if neighbor in path:
                continue
            g_new = g + self.distances[current][neighbor]
            f_new = g_new + self.heuristic_fn(neighbor, goal_idx)
            successors.append([neighbor, f_new, g_new])

        if not successors:
            return None, np.inf

        while True:
            successors.sort(key=lambda x: x[1])
            best = successors[0]
            print(successors)
            if best[1] > f_new:
                return None, best[1]
            alternative = successors[1][1] if len(successors) > 1 else np.inf
            result, returned_cost = self._rbfs(
                best[0], goal_idx, path + [best[0]], best[2], min(f_limit, alternative)
            )
            if result is not None:
                return result, returned_cost
            else:
                successors[0][1] = returned_cost


    def get_successors(self, node_idx):
        return [i for i in range(len(self.distances)) if self.distances[node_idx][i] != np.inf]

def heuristic(node_idx, goal_idx):
    # Straight-line heuristic: minimum edge cost from current node
    row = network.distances[node_idx]
    valid_edges = [row[i] for i in range(len(row)) if row[i] != np.inf and i != node_idx]
    # return min(valid_edges) if valid_edges else 0
    return 20


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
    plt.title("RBFS - Best Path Visualization")
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

    rbfs = RBFS(network, heuristic)
    path, cost = rbfs.search(start_idx, goal_idx)

    if path:
        path_names = [network.node_names[i] for i in path]
        print(f"\nBest path: {' -> '.join(path_names)}")
        print(f"Total cost: {cost:.2f}")
        plot_path(network, path)
    else:
        print("No path found.")
