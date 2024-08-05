
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency, X):
        # A_hat = A + I (Adding self-loops)
        A_hat = adjacency + torch.eye(adjacency.size(0)).to(adjacency.device)
        # D_hat
        D_hat = torch.diag_embed(torch.pow(A_hat.sum(1), -0.5))
        # Forward pass
        return F.relu(D_hat @ A_hat @ D_hat @ X @ self.linear.weight.T + self.linear.bias)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, adjacency, X):
        X = self.layer1(adjacency, X)
        return self.layer2(adjacency, X)
