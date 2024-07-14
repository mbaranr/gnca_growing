import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_3d_G(G: nx.Graph, 
                  coord: np.ndarray, 
                  title: str, 
                  angle: float=30.0,
                  node_size: float=100.0,
                  show_edges: bool=True,
                  transparent: bool=False):

    assert coord.shape[-1] == 3 and coord.ndim == 2   # making sure xyz is a 2d array and there are 3 dimensions (xyz)

    fig = plt.figure()
    fig.tight_layout()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.view_init(angle, angle)

    node_xyz = np.array([coord[v] for v in sorted(G)])
    edge_xyz = np.array([(coord[u], coord[v]) for u, v in G.edges()])

    ax.scatter(*node_xyz.T, s=node_size, ec='w', c='b')

    if show_edges:
        for edge in edge_xyz:
            ax.plot(*edge.T, color='black')

    if transparent:
        ax.grid(False)
        ax.set_axis_off() 

def E2G(E: np.ndarray,
        num_nodes: int=None):   
    """
    Converts an edge list to a NetworkX graph.

    Parameters:
        E (np.ndarray): List of edges.
        num_nodes (int): Number of nodes in G.
    
    Returns:
        G (nx.Graph): Networkx graph.
    """
    G = nx.Graph()
    for i in range(0 if num_nodes is None else num_nodes):
        G.add_node(i)
    for edge in E.T:
        G.add_edge(edge[0].item(), edge[1].item())
    return G

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
            