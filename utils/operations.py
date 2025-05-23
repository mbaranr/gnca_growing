import torch
import torch.nn as nn
import networkx as nx
import numpy as np

def normalize_adj(adj):
        """
        Row-normalize sparse matrix.
        """
        rowsum = adj.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        adj_normalized = torch.matmul(r_mat_inv, adj)

        return adj_normalized

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
        for edge in E:
                G.add_edge(edge[0].item(), edge[1].item())
        return G

def get_living_mask(x, adj, threshold=0.1):
        """
        Determines the 'alive' status of each node based on its and its neighbors' features.

        Parameters:
        x (torch.Tensor): Node features of shape (num_nodes, num_features).
        adj (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).
        threshold (float): Threshold value to determine if a node is alive.

        Returns:
        torch.Tensor: Boolean mask indicating alive status of each node.
        """
        # Extract the 'alive' feature, assumed to be the first feature in x
        alive_feature = x[:, :, 0].unsqueeze(2)
        
        # Propagate the alive feature to neighbors using adjacency matrix
        neighbor_alive_values = torch.matmul(adj, alive_feature)  # Shape: (num_nodes, 1)

        # Determine the max alive value in each node's neighborhood (including itself)
        return torch.max(alive_feature, neighbor_alive_values) > threshold

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    elif activation is None:
        return nn.Identity()  
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

