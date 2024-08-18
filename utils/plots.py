import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_3d_G(G: nx.Graph, 
              coord: np.ndarray, 
              title: str, 
              angle: float=30.0,
              node_size: float=100.0,
              show_edges: bool=True,
              transparent: bool=False,
              node_alpha: np.ndarray=None):

    assert coord.shape[-1] == 3 and coord.ndim == 2   # making sure xyz is a 2d array and there are 3 dimensions (xyz)

    fig = plt.figure()
    fig.tight_layout()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.view_init(angle, angle)

    # Use life_mask as the alpha values for nodes
    if node_alpha is None:
        node_alpha = np.ones(coord.shape[0])  # Default alpha is 1 for all nodes
    else:
        node_alpha = np.array(node_alpha).flatten()

    node_xyz = np.array([coord[v] for v in sorted(G)])

    # Plot all nodes with varying alpha values
    for i, xyz in enumerate(node_xyz):
        ax.scatter(*xyz, s=node_size, ec='w', c='b', alpha=node_alpha[i])

    if show_edges:
        # Plot edges with alpha = min(alpha of n1, alpha of n2)
        for u, v in G.edges():
            edge_alpha = min(node_alpha[u], node_alpha[v])
            edge_xyz = np.array([coord[u], coord[v]])
            ax.plot(*edge_xyz.T, color='black', alpha=edge_alpha)

    if transparent:
        ax.grid(False)
        ax.set_axis_off()

    plt.show()

def plot_3d_coords(coord: np.ndarray,
                   title: str, 
                   angle: float=30.0,
                   node_size: float=100.0,
                   transparent: bool=False):
    
    fig = plt.figure()
    fig.tight_layout()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.view_init(angle, angle)

    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=node_size, ec='w', c='b')

    if transparent:
        ax.grid(False)
        ax.set_axis_off()  
            