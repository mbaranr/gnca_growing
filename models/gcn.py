import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operations import normalize_adj, get_activation

# Adaptation from GeneralGNN spektral model (https://github.com/danielegrattarola/spektral/blob/master/spektral/models/general_gnn.py#L23)

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        layers=2,
        batch_norm=False,
        dropout=0.0,
        activation="relu",
        final_activation=None
    ):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential()
        
        for i in range(layers):
            layer_output = hidden_dim if i < layers - 1 else output_dim
            self.mlp.add_module(f"linear_{i}", nn.Linear(hidden_dim if i > 0 else input_dim, layer_output))
            
            if batch_norm:
                self.mlp.add_module(f"batch_norm_{i}", nn.BatchNorm1d(layer_output))
            
            if dropout > 0.0:
                self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))
            
            act_fn = activation if i < layers - 1 else final_activation
            self.mlp.add_module(f"activation_{i}", get_activation(act_fn))

    def forward(self, x):
        return self.mlp(x)
    
class GraphConv(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 dropout: float=0.0, 
                 bias: bool=True,
                 add_self: bool=False,
                 normalize_embedding: bool=True,
                 expect_normal: bool=False,
                 use_linear: bool=False  
                 ):
        super(GraphConv, self).__init__()

        self.expect_normal = expect_normal
        self.add_self = add_self
        self.dropout = dropout
        self.normalize_embedding = normalize_embedding
        self.use_linear = use_linear

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.bias = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None

        if use_linear:
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x, adj):  
        if not self.expect_normal:
            adj = normalize_adj(adj)

        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        # optional linear transformation
        if self.use_linear:
            x = self.linear(x)

        # graph convolution
        hidden = torch.matmul(adj, x)

        if self.add_self:
            hidden += x  

        if self.normalize_embedding:
            if hidden.dim() == 3:
                hidden = F.normalize(hidden, p=2, dim=2)
            elif hidden.dim() == 2:
                hidden = F.normalize(hidden, p=2, dim=1)

        return hidden


class GCNModel(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 message_passing=1, 
                 hidden_dim=256,
                 batch_norm=False,
                 hidden_activation="relu",
                 connectivity="cat",
                 dropout=0.0,
                 pool=None,
                 linear_conv=False
                 ):
        super(GCNModel, self).__init__()

        self.connectivity = connectivity
        self.pool = pool
        self.message_passing = message_passing
        
        self.pre = MLP(input_dim, hidden_dim, layers=2, batch_norm=batch_norm, dropout=dropout, activation=hidden_activation)

        self.gcs = nn.ModuleList()

        # message passing layers
        for i in range(message_passing):
            # dimension doubles after each layer if 'cat' is used
            dim = hidden_dim * (2 ** i) if self.connectivity == "cat" else hidden_dim
            self.gcs.append(GraphConv(dim, dim, dropout=dropout, expect_normal=False, add_self=False, use_linear=linear_conv))

        self.post = MLP(hidden_dim * (2 ** message_passing) if self.connectivity == "cat" else hidden_dim, output_dim, layers=2, batch_norm=batch_norm, dropout=dropout, activation=hidden_activation) 

        if self.connectivity == "sum":
            self.skip_connect = lambda x, skip: x + skip
        elif self.connectivity == "cat":
            self.skip_connect = lambda x, skip: torch.cat([x, skip], dim=-1)
        else:
            self.skip_connect = None

    def forward(self, x, adj):
        # pre-processing
        out = self.pre(x)
        skip = out
        
        # message passing
        for layer in self.gcs:
            z = layer(out, adj)
            if self.skip_connect is not None:
                out = self.skip_connect(z, skip)
                skip = out  # update skip connection for next layer
            else:
                out = z
        
        # pooling (for graph-level tasks)
        if self.pool == "sum":
            out = torch.sum(out, dim=1)
        elif self.pool == "mean":
            out = torch.mean(out, dim=1)
        elif self.pool == "max":
            out = torch.max(out, dim=1).values
        
        # post-processing 
        out = self.post(out)
        
        return out