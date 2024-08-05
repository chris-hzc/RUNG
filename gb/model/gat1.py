from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gb.model.fittable import StandardFittable
from gb.model.plumbing import Preprocess
from gb.model.tapeable import TapeableModule, TapeableParameter
from gb.preprocess import add_loops, gcn_norm
from gb.torchext import matmul
from gb.typing import Int, Float, IntSeq

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
features = None
classes = None
channels_in = None
channels_out = None


class GraphAttentionLayer(TapeableModule):
    def __init__(self, in_features: Int, out_features: Int, dropout: Float, alpha: Float, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = TapeableParameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        self.a = TapeableParameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, A: TensorType["batch_A": ..., "nodes", "nodes"], 
                     X: TensorType["batch_X": ..., "nodes", "channels_in"]) -> TensorType["batch_out": ..., "nodes", "channels_out"]:
        h = matmul(X, self.W)

        N = h.size(0)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        #attention = torch.where(A > 0, e, zero_vec)
        attention = A * e + (1 - A) * zero_vec
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = matmul(attention, h)

        if self.concat:
            return torch.nn.functional.elu(h_prime)
        else:
            return h_prime
        

@typechecked
class GAT(TapeableModule, StandardFittable):
    def __init__(self, n_feat: Int, n_class: Int, hidden_dims: IntSeq, n_heads: Int, dropout: Float = 0.5, alpha: Float = 0.2, concat: bool = True):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        self.concat = concat

        self.attention_layers = nn.ModuleList()
        in_channels = n_feat
        for hid_dim in hidden_dims:
            self.attention_layers.append(nn.ModuleList([GraphAttentionLayer(in_channels, hid_dim, dropout, alpha, concat) for _ in range(n_heads)]))
            in_channels = hid_dim * n_heads
        self.out_att = GraphAttentionLayer(in_channels, n_class, dropout, alpha, concat=False)

    def forward(self, A: TensorType["batch_A": ..., "nodes", "nodes"], 
                     X: TensorType["batch_X": ..., "nodes", "features"]) -> TensorType["batch_out": ..., "nodes", "classes"]:
        for layers in self.attention_layers:
            X = torch.cat([att(A, X) for att in layers], dim=-1)
            if self.concat:
                X = torch.nn.functional.dropout(X, self.dropout, training=self.training)
        X = self.out_att(A, X)
        return X






