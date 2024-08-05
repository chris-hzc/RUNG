import torch
from torch import nn

class MLP(nn.Module):

    def __init__(
            self,
            n_feat,
            n_class,
            hidden_dims,
            bias: bool = True,
            dropout: float = 0.5
    ):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias)
            for in_dim, out_dim in zip([n_feat] + hidden_dims, hidden_dims + [n_class])
        ])
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, X):
        for linear in self.linears[:-1]:
            X = self.dropout(nn.functional.relu(linear(X)))
        X = self.linears[-1](X)
        return X
