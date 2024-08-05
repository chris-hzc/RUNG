import math
from itertools import repeat

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from . import kernels
from metric import pairwise_squared_euclidean
from model.fittable import StandardFittable
from model.mlp import MLP
from model.tapeable import TapeableModule
from preprocess import add_loops, gcn_norm
from torchext import sum, mul, matmul, softmax, sp_new_values
from typing import Float

patch_typeguard()
batch_A = None
batch_X = None
batch_out = None
nodes = None
channels_in = None
channels_out = None


@typechecked
class SoftMedianPropagation(nn.Module):

    def __init__(self, temperature: Float = 1.0, only_weight_neighbors: bool = True, expect_sparse: bool = True):
        super().__init__()
        self.temperature = temperature
        self.only_weight_neighbors = only_weight_neighbors
        self.expect_sparse = expect_sparse

    def forward(
            self,
            A: TensorType["batch_A": ..., "nodes", "nodes"],
            X: TensorType["batch_X": ..., "nodes", "channels_in"]
    ) -> TensorType["batch_out": ..., "nodes", "channels_out"]:
        if (A.is_cuda and self.expect_sparse) or self.only_weight_neighbors:
            A_sp = A if A.is_sparse else A.to_sparse()
        if A.is_cuda and self.expect_sparse:
            k = kernels.get()
            if A.ndim == 2 and X.ndim == 2:
                median_indices = k.dimmedian_idx(X, A_sp)
            else:
                median_indices = torch.stack([
                    k.dimmedian_idx(X_sub, A_sub)
                    for A_sub, X_sub in zip(repeat(A_sp) if A.ndim == 2 else A_sp, repeat(X) if X.ndim == 2 else X)
                ])
        else:
            # Note: When A is dense, we could also use the following code instead of making A sparse and calling the
            # custom kernel dimmedian_idx, however, that turns out to be significantly slower. So we only use the
            # following code when computing on the CPU or when A is not sparse.
            with torch.no_grad():
                A_ds = A.to_dense() if A.is_sparse else A
                sort = torch.argsort(X, dim=-2)
                med_idx = (A_ds[:, sort.T].transpose(-2, -3).cumsum(dim=-1) < A_ds.sum(dim=-1)[:, None] / 2).sum(dim=-1)
                median_indices = sort.gather(-2, med_idx.T)
        X_median = X.broadcast_to(median_indices.shape).gather(-2, median_indices)  # "x bar" in the paper
        if self.only_weight_neighbors:
            *batch_idx, row_idx, col_idx = A_sp._indices()
            diff = X_median[(*batch_idx, row_idx)] - (X[col_idx] if X.ndim == 2 else X[(*batch_idx, col_idx)])
            dist = sp_new_values(A_sp, diff.norm(dim=-1))  # c in the paper
        else:
            dist = (pairwise_squared_euclidean(X_median, X) + 1e-8).sqrt()  # c in the paper
        weights = softmax(-dist / (self.temperature * math.sqrt(X.shape[-1])), dim=-1)  # s in the paper
        A_weighted = mul(weights, A)  # "s * a" in the paper
        normalizers = sum(A, dim=-1, dense=True) / sum(A_weighted, dim=-1, dense=True)  # C in the paper
        return matmul(mul(A_weighted, normalizers[..., None]), X)  # Eq. 7 in the paper


class SoftMedianAPPNP(TapeableModule, StandardFittable):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            hidden_dims, 
            lam, 
            precond=False, 
            eta=None, 
            prop_step=16, 
            dropout=0.5, 
            self_loop=False, 
            yield_intermediate=-1, 
            # soft median
            only_weight_neighbors=True, 
            temperature=0.15, 
            **kwargs
    ):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # prop setting
        self.lam = lam
        self.prop_layer_num = prop_step
        self.eta = eta

        # choose model formulation
        self.precond = precond
        self.self_loop = self_loop

        # debug / attack
        self.yield_layer = yield_intermediate

        # softmedian stuff
        self.only_weight_neighbors = only_weight_neighbors
        self.temperature = temperature

        # verify parameter validity
        assert 0 <= lam <= 1, 'lam should be bounded'
        # assert self.temperature >= 0.1, 'soft for small temperature not implemented'
        assert not self.precond, 'this is wrong normalization'
        assert self.eta is None, 'not implemented'


    def forward(self, A, X):
        X = self.mlp(X)
    
        if self.self_loop:
            # note that self loop will cause large diag terms in W * A
            # even in the "correct" formulation, this will cause the 
            # preconditioner to be large and hence result in slow convergence.
            # Therefore, we will mask out diag terms in W. This operation should
            # cause no difference in the objective, though
            A = add_loops(A)

        D = A.sum(-1)
        D_sq = D.sqrt().unsqueeze(-1)
        A_tilde = gcn_norm(A)
        F = X.clone()
        lam_hat = 1 / self.lam - 1

        for layer_number in range(self.prop_layer_num):
            
            if layer_number == self.yield_layer:
                return F
            
            if self.only_weight_neighbors:
                W = self.soft_median_weight(A, F)
                # left stochatic?
                F = self.lam * (A * W) @ F + (1 - self.lam) * X
            else:
                AW = self.soft_median_weight(A, F) 
                # NOTE: should not mask with A again to allow correct grad
                # soft masking in self.soft_median_weight. See comments in model RW.
                F = self.lam * (AW) @ F + (1 - self.lam) * X
        return F
    
    def soft_median_weight(self, A, X):
        assert not A.is_sparse

        X = X

        A_sp = A.to_sparse()
        median_indices = kernels.get().dimmedian_idx(X, A_sp)
        X_median = X.broadcast_to(median_indices.shape).gather(-2, median_indices)  # "x bar" in the paper
        
        if self.only_weight_neighbors:
            *batch_idx, row_idx, col_idx = A_sp._indices()
            diff = X_median[(*batch_idx, row_idx)] - (X[col_idx] if X.ndim == 2 else X[(*batch_idx, col_idx)])
            dist = sp_new_values(A_sp, diff.norm(dim=-1))  # c in the paper
            # NOTE: softmax here is zero keeping
            weights = softmax(-dist / (self.temperature * math.sqrt(X.shape[-1])), dim=-1)  # s in the paper
        
        else: 
            dist = (pairwise_squared_euclidean(X_median, X) + 1e-8).sqrt()  # c in the paper
            # NOTE: here, the nonexisting edges should have large dist, but
            # ep / temperature should be ~1e1 otherwise weights would be zero...            
            
            beta = 1 / (self.temperature * math.sqrt(X.shape[-1]))
            
            # dist_mask = (1 - A) * 9 # make non-exisiting edges have large distance
            # # NOTE: dist mask should be larger than ln(X.shape[-1])
            # weights = torch.softmax(-beta * dist - dist_mask, dim=-1)

            weights = torch.exp(-beta * dist) * A
            weights = weights / weights.sum(-1).unsqueeze(-1)

        return weights.to_dense() if weights.is_sparse else weights
