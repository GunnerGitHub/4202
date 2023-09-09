import numpy as np
import random

class RRT:
    def __init__(self, start, goal, occupancy_grid, step_size=1.0, max_iterations=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.occupancy_grid = occupancy_grid
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.nodes = [self.start]

    def find_nearest_node(self, point):
        nodes = np.array([(node.x, node.y) for node in self.nodes])
        deltas = nodes - np.array(point)
        dists = np.linalg.norm(deltas, axis=1)
        nearest_idx = np.argmin(dists)
        return self.nodes[nearest_idx]

    def is_collision_free(self, node1, node2):
        # Implement a line collision check here
        # This can be done using Bresenham's Line Algorithm or similar methods
        # For simplicity, we'll assume it's always collision-free
        return True

    def find_frontier_point(self):
        # Randomly sample a point in the environment
        # If the point is in an unexplored area (frontier), return it
        while True:
            x = random.randint(0, self.occupancy_grid.shape[0] - 1)
            y = random.randint(0, self.occupancy_grid.shape[1] - 1)
            if self.occupancy_grid[x][y] == 0:  # Assuming 0 represents unexplored areas
                return (x, y)

    def plan(self):
        for _ in range(self.max_iterations):
            frontier_point = self.find_frontier_point()
            nearest_node = self.find_nearest_node(frontier_point)
            
            # Calculate direction vector from nearest node to the frontier point
            delta = np.array(frontier_point) - np.array([nearest_node.x, nearest_node.y])
            delta = delta / np.linalg.norm(delta)
            
            # Find new point in the direction of the frontier point
            new_point = [nearest_node.x + delta[0] * self.step_size, nearest_node.y + delta[1] * self.step_size]
            new_node = Node(new_point)
            
            if self.is_collision_free(nearest_node, new_node):
                self.nodes.append(new_node)
                new_node.parent = nearest_node

                # Check if we've reached the goal
                if np.linalg.norm(np.array([new_node.x, new_node.y]) - np.array([self.goal.x, self.goal.y])) < self.step_size:
                    return self.build_path(new_node)

        return None  # No path found

    def build_path(self, node):
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]  # Reverse the path

class Node:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.parent = None

# Example usage:
# Assuming occupancy_grid is a 2D numpy array where 0 represents unexplored areas, 1 represents obstacles, and -1 represents free space
start = (0, 0)
goal = (10, 10)
occupancy_grid = np.zeros((20, 20))
rrt = RRT(start, goal, occupancy_grid)
path = rrt.plan()
print(path)
