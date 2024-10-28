import torch
import torch.nn.functional as F
import torch.nn as nn
from models.gcn import GCNModel
from utils.operations import get_living_mask

class GNCAModel(nn.Module):
    
    def __init__(self, input_dim, channel_n, fire_rate):
        super(GNCAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.gcn = GCNModel(input_dim, channel_n, connectivity="cat", hidden_dim=256, batch_norm=False, message_passing=1, linear_conv=False)

    def perceive(self, x, adj):
        # graph convolution to aggregate neighbor information
        neighbor_agg = torch.matmul(adj, x)

        # gradient-like information (difference between node state and neighbor aggregation)
        grad = neighbor_agg - x

        perception = torch.cat([x, grad], dim=-1)
        
        return perception

    def forward(self, x, adj, fire_rate=None):
        if fire_rate is None:
            fire_rate = self.fire_rate

        pre_life_mask = get_living_mask(x, adj)

        y = self.perceive(x, adj)

        dx = self.gcn(y, adj)

        update_mask = torch.rand_like(x[:, :, :1]) <= fire_rate

        x += dx * update_mask.float()

        # x[:, :, 0] = torch.sigmoid(x[:, :, 0]) # maintaining values between 0 and 1

        post_life_mask = get_living_mask(x, adj)

        life_mask = pre_life_mask & post_life_mask

        return x * life_mask.float()
