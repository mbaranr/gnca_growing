import numpy as np
import torch

def create_cubic_canvas(length: int, dist: float):
    """
    Creates a cube cloud of nodes and edges where each node is connected to its neighbors in a 3D grid.

    Parameters:
        length (int): Number of nodes along one dimension of the cube.
        dist (float): Distance between adjacent nodes.

    Returns:
        coord (np.ndarray): Array of shape (length^3, 3) containing the coordinates of the nodes.
        E (np.ndarray): Array of shape (num_edges, 2) containing the edges between nodes.
    """
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

    E = np.array(edges).T  # shape (2, num_edges)

    return coord, E

def create_cube_mask(length: int):
    return np.ones(length*length*length)

def create_pyramid_mask(length: int):

    mask = np.zeros((length, length, length), dtype=np.float32)

    # center of the base
    center = (length - 1) / 2

    for z in range(length):
        # calculate the base size for this z level
        base_size = length - z * 2  # base size decreases by 2 for each z level
        
        if base_size <= 0:
            break

        # boundaries of the square base at this level
        x_start = int(np.ceil(center - base_size / 2))
        x_end = int(np.floor(center + base_size / 2))
        y_start = x_start
        y_end = x_end
        
        mask[x_start:x_end+1, y_start:y_end+1, z] = 1

    return mask.flatten()

def create_sphere_mask(length: int, radius: int):

    mask = np.zeros((length, length, length), dtype=np.float32)

    # center of the grid
    center = (length - 1) / 2

    for x in range(length):
        for y in range(length):
            for z in range(length):
                # distance from the center of the sphere
                distance_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                
                if distance_from_center <= radius:
                    mask[x, y, z] = 1

    return mask.flatten()

def create_wall_mask(length: int, width: int):
    mask = np.zeros((length, length, length), dtype=np.float32)

    # center of the grid along the z-axis
    center_z = (length - 1) // 2

    z_start = max(0, center_z - width // 2)
    z_end = min(length, center_z + width // 2 + 1)

    mask[:, :, z_start:z_end] = 1

    return mask.flatten()