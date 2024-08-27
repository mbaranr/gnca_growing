import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_3d_G(G: nx.Graph, 
              coord: np.ndarray, 
              title: str, 
              ax=None,
              angle: float=30.0,
              node_size: float=100.0,
              show_edges: bool=True,
              transparent: bool=False,
              node_alpha: np.ndarray=None,
              fixed_range: bool=False,
              axis_range: float=None):
    
    assert coord.shape[-1] == 3 and coord.ndim == 2   # making sure xyz is a 2d array and there are 3 dimensions (xyz)

    if ax is None:
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111, projection="3d")

    ax.set_title(title)
    ax.view_init(angle, angle)

    if node_alpha is None:
        node_alpha = np.ones(coord.shape[0])  # Default alpha is 1 for all nodes
    else:
        node_alpha = np.array(node_alpha).flatten()

    node_xyz = np.array([coord[v] for v in sorted(G)])

    significant_nodes = node_alpha > 0

    significant_coords = coord[significant_nodes]
    significant_alpha = node_alpha[significant_nodes]

    # Plot significant nodes
    ax.scatter(significant_coords[:, 0], significant_coords[:, 1], significant_coords[:, 2], 
               s=node_size, ec='w', c='b', alpha=significant_alpha)

    if show_edges:
        # Plot edges with alpha = min(alpha of n1, alpha of n2)
        for u, v in G.edges():
            edge_alpha = min(node_alpha[u], node_alpha[v])
            edge_xyz = np.array([coord[u], coord[v]])
            ax.plot(*edge_xyz.T, color='black', alpha=edge_alpha)

    if transparent:
        ax.grid(False)
        ax.set_axis_off()

    if fixed_range:
        if axis_range is None:
            # Compute axis range from the data
            min_range = np.min(coord)
            max_range = np.max(coord)
            axis_range = max(max_range - min_range, 1.0)  # Ensure non-zero range
        ax.set_xlim([0, axis_range])
        ax.set_ylim([0, axis_range])
        ax.set_zlim([0, axis_range])
        
    return ax

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

def plot_loss(losses, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

def generate_pool_image(coords, pool, epoch, num_graphs=50, filename='pool_'):

    grid_size = (10, 5)  # 10x5 grid for 50 graphs

    fig = plt.figure(figsize=(grid_size[1] * 4, grid_size[0] * 4))
    axes = [fig.add_subplot(grid_size[0], grid_size[1], i + 1, projection='3d') for i in range(num_graphs)]
    
    for i in range(num_graphs):
        ax = axes[i]
        node_alpha = pool.x[i][:, 0].cpu().numpy()
        plot_pool_graph(ax, coords, node_alpha)
    
    plt.tight_layout()
    plt.savefig('train_log/' + filename + str(epoch) + '.png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def plot_pool_graph(ax, coords, node_alpha, axis_range=1, angle=30, transparent=True):
    """
    Plot a single graph on a given axis.
    """
    # Clear the axis
    ax.clear()

    ax.view_init(angle, angle)

    if transparent:
        ax.grid(False)
        ax.set_axis_off()

    # ax.set_xlim([0, axis_range])
    # ax.set_ylim([0, axis_range])
    # ax.set_zlim([0, axis_range])
    
    significant_nodes = node_alpha > 0
    
    significant_coords = coords[significant_nodes]

    if significant_coords.size > 0:
        ax.scatter(significant_coords[:, 0], significant_coords[:, 1], s=10, c='b', alpha=node_alpha[significant_nodes])
