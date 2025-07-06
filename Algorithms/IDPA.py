import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class IDAStar:
    def __init__(self, network, heuristic_fn):
        self.network = network
        self.distances = network.distances
        self.heuristic_fn = heuristic_fn

    def search(self, start_idx, goal_idx):
        threshold = self.heuristic_fn(start_idx, goal_idx)
        path = [start_idx]

        while True:
            temp = self._search(path, 0, threshold, goal_idx, set())
            if isinstance(temp, list):
                return temp
            if temp == np.inf:
                return None
            threshold = temp

    def _search(self, path, g, threshold, goal_idx, visited):
        current = path[-1]
        f = g + self.heuristic_fn(current, goal_idx)

        if f > threshold:
            return f
        
        if self.is_goal(current, goal_idx):
            return list(path)


        min_threshold = np.inf
        visited.add(current)

        for neighbor in self._get_successors(current, visited):
            path.append(neighbor)
            temp = self._search(path, g + self.distances[current][neighbor], threshold, goal_idx, visited)
            if isinstance(temp, list):
                return temp
            if temp < min_threshold:
                min_threshold = temp
            path.pop()

        visited.remove(current)
        return min_threshold

    def _get_successors(self, node_idx, visited):
        return [i for i in range(len(self.distances))
                if self.distances[node_idx][i] != np.inf and i not in visited]
    
    def is_goal(self, current_idx, goal_idx):
        return current_idx == goal_idx


def heuristic(node_idx, goal_idx):
    row = network.distances[node_idx]
    valid_edges = [row[i] for i in range(len(row)) if row[i] != np.inf and i != node_idx]
    return min(valid_edges)+4 if valid_edges else 1

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

    plt.title("IDA* - Best Path Visualization")
    plt.show()


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

def compute_idastar_complexity(graph_size, branching_factor, goal_depth):
    """
    Returns estimated time and space complexity strings for IDA* search.
    """
    time_complexity = f"O({branching_factor}^{goal_depth})"
    space_complexity = f"O({goal_depth})"
    return time_complexity, space_complexity

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

    ida_star = IDAStar(network, heuristic)
    best_path = ida_star.search(start_idx, goal_idx)

    if best_path:
        cost = sum(network.distances[best_path[i]][best_path[i + 1]] for i in range(len(best_path) - 1))
        path_names = [network.node_names[i] for i in best_path]
        print(f"\nBest path: {' -> '.join(path_names)}")
        print(f"Total cost: {cost:.2f}")
        plot_path(network, best_path)
    else:
        print("No path found.")
# Estimate complexity after search
    estimated_b = 2  # rough average successors per node (can be estimated)
    estimated_d = len(best_path) - 1 if best_path else 0

    tc, sc = compute_idastar_complexity(graph_size=len(node_names), branching_factor=estimated_b, goal_depth=estimated_d)
    print("\n--- Theoretical Complexity (IDA*) ---")
    print("Estimated Branching Factor (b):", estimated_b)
    print("Estimated Depth to Goal (d):", estimated_d)
    print(f"Time Complexity:  {tc}")
    print(f"Space Complexity: {sc}")
