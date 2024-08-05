import torch



# Graph related

def add_loops(A):
    n = A.shape[-1]
    return A + torch.eye(n, device=A.device)

def sym_norm(A):
    Dsq = A.sum(-1).sqrt()
    return A / Dsq / Dsq.unsqueeze(-1)



# metric

def accuracy(
        scores,
        y_true
):
    return (scores.argmax(dim=-1) == y_true).count_nonzero(dim=-1) / y_true.shape[0]

# python helper

def sub_dict(dct, *filter_keys, optional=False):
    # Note: This method raises a KeyError if a desired key is not found, and that is exactly what we want.
    if not optional:
        return {key: dct[key] for key in filter_keys}
    else:
        return {key: dct[key] for key in dct if key in filter_keys}

# tensor helper 

def sp_new_values(t, values):
    out = torch.sparse_coo_tensor(t._indices(), values, t.shape)
    # If the input tensor was coalesced, the output one will be as well since we don't modify the indices.
    if t.is_coalesced():
        with torch.no_grad():
            out._coalesced_(True)
    return out


# model helper

def pairwise_squared_euclidean(X, Y):
    '''
    Adapted from [are_gnn_robust](https://github.com/LoadingByte/are-gnn-defenses-robust)

    $$
    Z_{ij} = \sum_k (F_{ik} - F_{jk})^2 \
        = \sum_k F_{ik}^2 + F_{jk}^2 - 2  F_{ik}  F_{jk}, 
    $$
    where $\sum_k F_{ik}  F_{jk} = (F F^\top)_{ij}$
    The matmul is already implemented efficiently in torch
    '''

    squared_X_feat_norms = (X * X).sum(dim=-1)  # sxfn_i = <X_i|X_i>
    squared_Z_feat_norms = (Y * Y).sum(dim=-1)  # szfn_i = <Z_i|Z_i>
    pairwise_feat_dot_prods = X @ Y.transpose(-2, -1)  # pfdp_ij = <X_i|Z_j> # clever...
    return (-2 * pairwise_feat_dot_prods + squared_X_feat_norms[:, None] + squared_Z_feat_norms[None, :]).clamp_min(0)
