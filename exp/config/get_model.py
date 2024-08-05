# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from utils import accuracy
import copy
import yaml
from exp.result_io import save_acc, rep_save_model
from typeguard import typechecked

# models
# from model.softmedian import SoftMedianAPPNP, SoftMedianPropagation
from model.mlp import MLP
from model.att_func import get_l12_att_func, get_default_att_func, get_log_att_func, get_mask_att_func, get_step_p_norm_att_func, get_soft_step_l21_att_func, get_mcp_att_func, get_scad_att_func
from model.rung import RUNG


# preprocessing for sontructing models
# from model.plumbing import GraphSequential, PreprocessA, PreprocessX
from collections import OrderedDict

# computation pkgs
import torch
from torch import nn
import numpy as np

import re
import gb

def get_model_default(
    dataset, model_name, custom_model_params={}, custom_fit_params={}, as_paper=True, seed=None, D=None, device='cuda:0'
):
    torch.manual_seed(0 if seed is None else seed)

    A, X, y = get_dataset(dataset)
    sp = get_splits(y)
    train_idx, val_idx, test_idx = sp[0]

    D = X.shape[1] if D is None else D
    C = y.unique().shape[0]

    if model_name == 'RUNG':
        return RUNG(D, C, [64], get_mcp_att_func(custom_model_params['gamma']), 0.9).to(device), custom_fit_params
    elif model_name == 'GCN':
        return gb.model.GCN(n_feat=D, n_class=C, hidden_dims=[64], dropout=0.5).to(device), custom_fit_params
    elif model_name == 'GAT':
        return gb.model.GAT(num_of_layers=2, num_heads_per_layer=[4,1], num_features_per_layer=[D,16,C], add_skip_connection=True, bias=True,
                 dropout=0.1, log_attention_weights=False).to(device), custom_fit_params
    elif model_name in ['APPNP','L1']:
        return RUNG(D, C, [64], get_l12_att_func(custom_model_params['norm']), 0.9).to(device), custom_fit_params
    elif model_name == 'MLP':
        return RUNG(D, C, [64], get_mcp_att_func(custom_model_params['gamma']), 0.9, prop_step=0).to(device), custom_fit_params
        
