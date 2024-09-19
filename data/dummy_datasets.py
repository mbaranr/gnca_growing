import numpy as np
import torch

def create_3d_canvas(length: int, dist: float):
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

    E = np.array(edges).T  # shape (num_edges, 2)

    return coord, E

def create_2d_canvas(length: int, dist: float):
    """
    Creates a square cloud of nodes and edges where each node is connected to its neighbors in a 2D grid.

    Parameters:
        length (int): Number of nodes along one dimension of the plain.
        dist (float): Distance between adjacent nodes.

    Returns:
        coord (np.ndarray): Array of shape (length^2, 2) containing the coordinates of the nodes.
        E (np.ndarray): Array of shape (num_edges, 2) containing the edges between nodes.
    """
    values = torch.linspace(0, 1, steps=length) * dist
    coord = torch.stack(torch.meshgrid(values, values, indexing='xy')).reshape(2, -1).T.numpy()

    edges = []
    for x in range(length):
        for y in range(length):
            node_idx = x * length + y
            if x < length - 1:  
                edges.append((node_idx, node_idx + length))
            if y < length - 1:  
                edges.append((node_idx, node_idx + 1))

    E = np.array(edges)  # shape (num_edges, 2)

    return coord, E

def create_square_mask(length: int):
    return np.ones(length*length)

from PIL import Image
import numpy as np

def create_image_mask(image_path: str, length: int):
    """
    Creates a binary mask from a PNG image where all pixels with non-zero alpha (non-transparent)
    are set as 1 in the mask, and transparent pixels are set to 0.
    """
    img = Image.open(image_path).convert("RGBA")  # ensure RGBA image
    img = img.resize((length, length))

    alpha_channel = np.array(img)[:, :, 3]  # alpha channel (4th channel)
    alpha_channel = np.flipud(alpha_channel)
    
    mask = (alpha_channel > 0).astype(np.float32)
    mask = mask.flatten()

    return mask

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

def create_cuboid_mask(length: int, width: int):
    mask = np.zeros((length, length, length), dtype=np.float32)

    # center of the grid along the z-axis
    center_z = (length - 1) // 2

    z_start = max(0, center_z - width // 2)
    z_end = min(length, center_z + width // 2 + 1)

    mask[:, :, z_start:z_end] = 1

    return mask.flatten()

