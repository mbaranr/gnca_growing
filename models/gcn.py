import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.operations import normalize_adj

# Adaptation from GeneralGNN spektral model (https://github.com/danielegrattarola/spektral/blob/master/spektral/models/general_gnn.py#L23) (https://github.com/danielegrattarola/GNCA/blob/master/models/gnn_ca_simple.py)

class GCNModel(nn.Module):

    def __init__(self, 
                 activation=1, 
                 message_passing=1, 
                 hidden=256,
                 batch_norm=False,
                 hidden_activation="relu",
                 connectivity="cat",
                 aggregate="sum", 
                 device="cuda"
                 ):
        super(GCNModel, self).__init__()
        
        # self.pre = 


def MLP(nn.Module):
    def __init__(self, 
                 output,
                 hidden=256,
                 layers=2,
                 batch_norm=False,
                 dropout=0.0,
                 activation="relu",
                 final_activation=None
                 ):
        self.hidden = hidden
        self.output = output
        
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output))
            # Batch norm
            if self.batch_norm:
                self.mlp.add(BatchNormalization())
            # Dropout
            self.mlp.add(Dropout(self.dropout_rate))
            # Activation
            self.mlp.add(get_act(activation if i < layers - 1 else final_activation))





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