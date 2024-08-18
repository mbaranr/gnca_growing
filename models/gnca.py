import sys
sys.path.append('../')  

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.graph_convolution import GraphConv
from utils.operations import get_living_mask

class GNCAModel(nn.Module):
    
    def __init__(self, input_dim, channel_n, fire_rate):
        super(GNCAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        self.relu = nn.LeakyReLU
        self.gc1 = GraphConv(input_dim, 128)
        self.gc2 = GraphConv(128, channel_n)

    def perceive(self, x, adj):
    
        # Graph convolution to aggregate neighbor information
        neighbor_agg = torch.matmul(adj, x)

        # Compute gradient-like information (difference between node state and neighbor aggregation)
        grad = neighbor_agg - x
        
        # Concatenate the original states with the gradient information
        perception = torch.cat([x, grad], dim=-1)
        
        return perception

    def forward(self, x, adj, fire_rate=None, angle=0.0, step_size=1.0):
        if fire_rate is None:
            fire_rate = self.fire_rate

        pre_life_mask = get_living_mask(x, adj)

        y = self.perceive(x, adj)

        dx = self.gc1(y, adj)
        dx = self.relu(dx)
        dx = self.gc2(dx, adj)

        update_mask = torch.rand_like(x[:, :, :, :1]) <= fire_rate
        x += dx * update_mask.float()

        post_life_mask = get_living_mask(x, adj)

        life_mask = pre_life_mask & post_life_mask

        return x * life_mask.float()
