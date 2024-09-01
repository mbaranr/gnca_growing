import sys
sys.path.append('../')  

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operations import normalize_adj

class GraphConv(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 dropout: float=0.0, 
                 bias: bool=True,
                 add_self: bool=False,
                 normalize_embedding: bool=True,
                 expect_normal: bool=False,
                 device: torch.device=None):
               
        super(GraphConv, self).__init__()

        self.expect_normal = expect_normal
        self.add_self = add_self
        self.dropout = dropout
        self.normalize_embedding = normalize_embedding

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):  
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        if not self.expect_normal:
            adj = normalize_adj(adj)

        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        hidden = self.linear(x)  # linear transformation
        hidden = torch.matmul(adj, hidden)  # aggregation

        if self.add_self:
            hidden += x  

        if self.bias is not None:
            hidden += self.bias

        if self.normalize_embedding:
            if hidden.dim() == 3:
                hidden = F.normalize(hidden, p=2, dim=2)
            elif hidden.dim() == 2:
                hidden = F.normalize(hidden, p=2, dim=1)

        return hidden
    
