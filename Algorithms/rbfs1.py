import math

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

def heuristic(node, goal):
    """Simple heuristic: straight-line (if available), else zero."""
    # You can customize this!
    return 0

def rbfs(node, goal, g, f_limit, path, visited):
    if node == goal:
        return path, g

    # Generate successors
    successors = []
    for neighbor, cost in enumerate(distance_matrix[node]):
        if cost > 0 and neighbor not in visited:
            h = heuristic(neighbor, goal)
            f = g + cost + h
            successors.append((f, neighbor, cost))

    if not successors:
        return None, math.inf

    # Sort successors by f value
    successors.sort(key=lambda x: x[0])

    while successors:
        best_f, best_node, best_cost = successors[0]
        if best_f > f_limit:
            return None, best_f

        # Next best alternative
        alternative = successors[1][0] if len(successors) > 1 else math.inf

        result, best_f_new = rbfs(
            best_node, goal, g + best_cost, min(f_limit, alternative),
            path + [best_node], visited | {best_node}
        )
        if result is not None:
            return result, best_f_new

        # Update best_f in successors
        successors[0] = (best_f_new, best_node, best_cost)
        successors.sort(key=lambda x: x[0])

    return None, math.inf

# Example: Find path from node 0 to node 7
start = 0
goal = 6
path, cost = rbfs(start, goal, 0, math.inf, [start], {start})
print(f"Best path: {path}")
print(f"Total cost: {cost}")
