import torch

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