"""RUNG: Robust UNbiased aggregation for Graph neural networks.

Reference implementation of Eq. (8) in

    Hou, Feng, Derr & Liu. "Robust Graph Neural Networks via Unbiased Aggregation".
    NeurIPS 2024. https://arxiv.org/abs/2311.14934

Each propagation layer performs one Quasi-Newton IRLS (QN-IRLS) step on the
Robust and Unbiased Graph signal Estimator (RUGE):

    min_F  sum_{(i,j) in E} rho_gamma(|| f_i / sqrt(d_i) - f_j / sqrt(d_j) ||_2)
           + lambda * sum_i || f_i - f_i^(0) ||_2^2,

which unrolls into the layer

    F^(k+1) = (diag(q^(k)) + lambda I)^{-1} ((W^(k) * A_tilde) F^(k) + lambda F^(0)),

where W_ij = d rho(y) / d y^2 evaluated at y_ij = || f_i / sqrt(d_i) - f_j / sqrt(d_j) ||_2,
and q_m = sum_j W_mj A_mj / d_m.
"""
import torch
from torch import nn

from model.mlp import MLP
from utils import add_loops, pairwise_squared_euclidean, sym_norm



class RUNG(nn.Module):
    """Decoupled RUNG model: an MLP encoder followed by `prop_step` robust
    unbiased aggregation layers (APPNP-style, see Sec. 3.3 of the paper).

    Args:
        in_dim: input feature dimension.
        out_dim: number of classes.
        hidden_dims: hidden sizes of the MLP encoder.
        w_func: edge reweighting function W = d rho(y) / d y^2, applied to the
            *squared* normalized feature distance (see `model/att_func.py`).
        lam_hat: lambda_hat = 1 / (1 + lambda) in [0, 1]; controls the strength
            of the skip connection to F^(0).
        quasi_newton: use QN-IRLS (Eq. 7, stepsize-free) if True, otherwise
            first-order IRLS (Eq. 6) with stepsize `eta`.
        eta: IRLS stepsize; must be None when `quasi_newton=True`.
        prop_step: number of aggregation layers K.
        dropout: dropout rate in the MLP encoder.
    """

    def __init__(
            self, 
            in_dim: int, 
            out_dim: int, 
            hidden_dims: list[int], 
            w_func: callable, 
            lam_hat: float, 
            quasi_newton=True, 
            eta=None, 
            prop_step=10, 
            dropout=0.5, 
    ):
        super().__init__()
        # MLP Settings (decoupled architecture: F = RUNG(MLP(A, F0)))
        self.mlp = MLP(in_dim, out_dim, hidden_dims, dropout=dropout)

        # Graph Smoothing Settings
        # objective: \sumedge |fi - fj| + \sumnode \lambda |fi - fi0|
        self.lam_hat = lam_hat 
        # variable substitution: lam_hat = 1 / (1 + lam), s.t. lam_hat
        # is bounded in [0, 1]
        self.lam = 1 / lam_hat - 1 
        self.quasi_newton = quasi_newton
        self.prop_layer_num = prop_step
        self.w: callable = w_func # W = d_{y^2} \rho(y)
        self.eta = eta

        # Verify Parameter Validity
        assert 0 <= lam_hat <= 1, 'lam_hat should be in [0, 1]!'
        if quasi_newton:
            assert eta is None, 'no need to specify stepsize in QN-IRLS'
        else:
            assert 0 < eta, 'must use nonzero stepsize'
    

    
    def forward(self, A, F):
        # decoupled architecture: F = RUNG(MLP(A, F0))
        F0 = self.mlp(F)

        # add self loop to graph to avoid zero degree
        A = add_loops(A)
        # record degree matrix
        D = A.sum(-1)
        D_sq = D.sqrt().unsqueeze(-1)
        # normalize A
        A_tilde = sym_norm(A)
        
        # record F0 for skip connection (teleportation in APPNP)
        F = F0

        for layer_number in range(self.prop_layer_num):
            # Z_{ij} = |f_i - f_j|_2^2
            Z = pairwise_squared_euclidean(F / D_sq, F / D_sq).detach()
            # W_{ij} = d_{y^2} \rho(y), y = |f_i - f_j|_2
            W = self.w(Z)
            # diag terms in W set to zero: see Remark 2 in paper
            W[torch.arange(W.shape[0]), torch.arange(W.shape[0])] = 0
            # guard against NaNs in W (e.g. from 0/0 in rho)
            W[torch.isnan(W)] = 1
            
            if self.quasi_newton: # Quasi-Newton IRLS, Eq. (7)/(8)
                # approx Hessian
                Q_hat = ((W * A).sum(-1) / D + self.lam).unsqueeze(-1)
                # Unbiased Robust Aggregation: guaranteed convergence!
                F = (W * A_tilde) @ F / Q_hat + self.lam * F0 / Q_hat
            
            else: # first-order IRLS, Eq. (6)
                diag_q = torch.diag((W * A).sum(-1)) / D    
                # gradient of H_hat
                grad_smoothing = 2 * (diag_q - W * A_tilde) @ F
                grad_reg = 2 * (self.lam * F - self.lam * F0)
                F = F - self.eta * (grad_smoothing + grad_reg)

        return F

