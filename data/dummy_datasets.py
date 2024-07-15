import numpy as np
import torch

def create_cube(length: int, dist: float):
    """
    Creates a cube cloud of nodes and edges where each node is connected to its neighbors in a 3D grid.

    Parameters:
        length (int): Number of nodes along one dimension of the cube.
        dist (float): Distance between adjacent nodes.

    Returns:
        coord (np.ndarray): Array of shape (length^3, 3) containing the coordinates of the nodes.
        E (np.ndarray): Array of shape (num_edges, 2) containing the edges between nodes.
    """
    # Create coordinates
    values = torch.linspace(0, 1, steps=length) * dist
    coord = torch.stack(torch.meshgrid(values, values, values, indexing='xy')).reshape(3, -1).T.numpy()

    edges = []
    for x in range(length):
        for y in range(length):
            for z in range(length):
                node_idx = x * length * length + y * length + z
                if x < length - 1:  
                    edges.append((node_idx, node_idx + length * length))
                if y < length - 1:  
                    edges.append((node_idx, node_idx + length))
                if z < length - 1:  
                    edges.append((node_idx, node_idx + 1))

    E = np.array(edges).T  # Shape (2, num_edges)

    return coord, E
