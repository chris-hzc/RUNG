import os

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import mode
from sklearn.model_selection import train_test_split

import numpy as np
import torch

def get_dataset(name):
    '''Returns respective dataset 
    
    Returns:
        A: adjacency matrix. sparse or dense
        X: node feature matrix
        train_idx: training node idx
        val_idx: validation node idx
        test_idx: testing node idx
    '''


def get_datasplit(name):
    '''Generate the same training data split for the same dataset'''
    return


def get_target_node_idx(name):
    '''Generate the same target node split for the same dataset.'''
    return


def get_dataset(dataset_name: str):
    if dataset_name in ("citeseer", "cora"):
        try:
            return _load_npz(
                os.path.join(
                    os.path.dirname(__file__), "..", "data", dataset_name + ".npz"
                )
            )
        except FileNotFoundError as e:
            # Fallback for runs via SEML on the GPU cluster.
            raise e
            return _load_npz(f"{FALLBACK_SRC_PATH}/data/{dataset_name}.npz")
    elif dataset_name in ['flickr', 'reddit','dblp','pubmed', 'polblogs','acm','BlogCatalog','uai']:
        A = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "adj.pt"))
        X = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "fea.pt")).to(torch.float32)
        y = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", dataset_name, "label.pt"))
        
        return A.cuda(), X.cuda(), y.cuda()
    else:
        A = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "adj.pt"))
        X = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "fea.pt")).to(torch.float32)
        y = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "heter_data", dataset_name, "label.pt"))
        
        return A.cuda(), X.cuda(), y.cuda()


def _load_npz(path: str):
    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)
        A = _fix_adj_mat(_extract_csr(loader, "adj"))
        _, comp_ids = connected_components(A)
        lcc_nodes = np.nonzero(comp_ids == mode(comp_ids)[0])[0]
        A = torch.tensor(A[lcc_nodes, :][:, lcc_nodes].todense(), dtype=torch.float32)
        if "attr_data" in loader:
            X = torch.tensor(
                _extract_csr(loader, "attr")[lcc_nodes, :].todense(),
                dtype=torch.float32,
            )
        else:
            X = torch.eye(A.shape[0])
        if "labels" in loader:
            y = torch.tensor(loader["labels"][lcc_nodes], dtype=torch.int64)
        else:
            y = None
        return A.cuda(), X.cuda(), y.cuda()


def _extract_csr(loader, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            loader[f"{prefix}_data"],
            loader[f"{prefix}_indices"],
            loader[f"{prefix}_indptr"],
        ),
        loader[f"{prefix}_shape"],
    )


def _fix_adj_mat(A: sp.csr_matrix) -> sp.csr_matrix:
    # Some adjacency matrices do have some values on the diagonal, but not everywhere. Get rid of this mess.
    A = A - sp.diags(A.diagonal())
    # For some reason, some adjacency matrices are not symmetric. Fix this following the Nettack code.
    A = A + A.T
    A[A > 1] = 1
    return A


def get_splits(
    y, 
    more_sps=0,
):
    """
    Produces 5 deterministic 10-10-80 splits.
    """
    if more_sps != 0:
        return [
            _three_split(y.cpu(), 0.1, 0.1, random_state=r)
            for r in [1234, 2021, 1309, 4242, 1698] + list(range(more_sps))
        ]
    return [
        _three_split(y.cpu(), 0.1, 0.1, random_state=r)
        #for r in [1234, 2021, 1309, 4242, 1698]
        for r in [1534, 2021, 1323, 1535, 1698]
    ]
    

def _three_split(
    y, size_1, size_2, random_state
):
    idx = np.arange(y.shape[0])
    idx_12, idx_3 = train_test_split(
        idx, train_size=size_1 + size_2, stratify=y, random_state=random_state
    )
    idx_1, idx_2 = train_test_split(
        idx_12,
        train_size=size_1 / (size_1 + size_2),
        stratify=y[idx_12],
        random_state=random_state,
    )
    return torch.tensor(idx_1), torch.tensor(idx_2), torch.tensor(idx_3)
