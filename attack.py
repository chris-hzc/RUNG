
# append path
import sys

sys.path.append("./")
import os
# dataset, train and eval
from train_eval_data.get_dataset import get_dataset, get_splits
from gb.exutil import make_attacked_model
from gb.metric import accuracy
from train_eval_data.fit import fit
import copy
import yaml

from exp.config.get_model_cora import get_model_default_cora
from exp.config.get_model_citeseer import get_model_default_citeseer
from exp.config.get_model import get_model_default
from exp.result_io import save_rep_edge_flips, load_rep_edge_flips, save_acc, rep_load_model, rep_save_model

# typing
from typing import Sequence
from torchtyping import TensorType

# computation pkgs
import torch
from torch import nn
import numpy as np

# attacks
from gb.attack.gd import proj_grad_descent, greedy_grad_descent
from gb.metric import margin
from gb.pert import edge_diff_matrix

import optuna
import argparse

path = ""

parser = argparse.ArgumentParser(description='Train classification network')
# model setting
parser.add_argument('--model',type=str, default='GCN')
parser.add_argument('--norm',type=str, default='MCP')
parser.add_argument('--gamma',type=float, default=3.0)
parser.add_argument('--data',type=str, default='squirrel')

args = parser.parse_args()
if args.model == 'APPNP':
    args.norm = 'L2'
elif args.model == 'L1':
    args.norm = 'L1'


#from my_exp.tuning_setting_files.every_budget_cora_global import tune_every_budget_mcp, tune_every_budget_softmedian, tune_every_budget_twirls

def make_A_pert(A, flip):
    return A + edge_diff_matrix(flip, A)

def eval_evasion(model, A_pert, X, y, test_idx):
    model.eval()
    return accuracy(model(A_pert, X)[test_idx, :], y[test_idx]).item()



def global_evasion_pgd(
    attacked_model: nn.Module, A, X, y, test_idx, budget_edge_num: int, init_A=None, model_atk=None, **kwargs
):
    # unpacking args
    attacked_model.eval()
    # special case for svd
    # setup grad func
    def loss_fn(flip):
        if model_atk == 'svd_gcn':
            pert = (1 - 2 * A) * flip * eigenspace_proj # rank = 50
            model_layers = attacked_model.modules()
            next(model_layers)
            A_svd = next(model_layers)(A)
            out = next(model_layers)(A_svd + pert, X)
        else:
            A_pert = A + (flip * (1 - 2 * A))
            out = attacked_model(A_pert, X)
        return margin(out[test_idx, :], y[test_idx]).tanh().mean()

    def grad_fn(flip):
        loss = loss_fn(flip)
        return torch.autograd.grad(loss, flip)[0]

    # PGD
    if init_A is None:
        flip, loss = proj_grad_descent(
            A.shape, True, A.device, budget_edge_num, grad_fn, loss_fn, grad_clip=1, **kwargs
        )
    else:
        flip, loss = proj_grad_descent(
            init_A, True, A.device, budget_edge_num, grad_fn, loss_fn, **kwargs
        )
    acc = eval_evasion(attacked_model, make_A_pert(A, flip), X, y, test_idx)
    clean = eval_evasion(attacked_model, A, X, y, test_idx)
    return clean, acc, flip

def rep_global_evasion(
    dataset_name, model, fit_params, attack_method: callable, budget_ratio: float, return_model=False, seed=None, init_As=None, iter=200, **params
):
    # prepare model
    A, X, y = get_dataset(dataset_name)
    sps = get_splits(y)

    cleans, accs, edge_flips, models = [], [], [], []
    for i, (train_idx, val_idx, test_idx) in enumerate(sps):
        model_cur_rep = copy.deepcopy(model)
        torch.manual_seed(seed if seed is not None else 0)

        #model_cur_rep.fit((A, X), y, train_idx, val_idx, **fit_params)
        A, X, y = A.cuda(), X.cuda(), y.cuda()
        #fit(model_cur_rep, A, X, y, train_idx, val_idx, **fit_params)
        
        model_path = path+f'exp/models/{dataset_name}/{f"{args.model}_{args.norm}_{args.gamma}"}/0.000/split_0/rand_model_{i}/clean_model'
 
        model_cur_rep.load_state_dict(torch.load(model_path))
        
        
        budget_edge_num = int(budget_ratio * A.count_nonzero().item() // 2)
        
        clean, acc, edge_flip = attack_method(
            model_cur_rep, A, X, y, test_idx, budget_edge_num, 
            init_A=make_A_pert(torch.zeros_like(A), init_As[i]) if init_As is not None else None, 
            iterations=iter, 
            **params
        )
        models.append(model_cur_rep)
        accs.append(acc)
        cleans.append(clean)
        edge_flips.append(edge_flip)

        print(clean,acc)

    if return_model:        
        return cleans, accs, edge_flips, models
        
    return cleans, accs, edge_flips

def rep_transfer_evasion(dataset_name, trained_transfer_to_models, edge_flips, rep_per_split=5):
    '''
    model: transfer to
    edge_flip: N by 2 matrix
    '''
    A, X, y = get_dataset(dataset_name)
    sps = get_splits(y)
    assert len(sps) * rep_per_split == len(trained_transfer_to_models), 'Model number does not match split and rep number. Now transfer requires pert to avg on other initializations of a model'

    accs = []
    for model_id, ((_, _, test_idx), edge_flip) in enumerate(zip(sps, edge_flips)):
        for i in range(rep_per_split):
            model = trained_transfer_to_models[model_id * rep_per_split + i]
            accs.append(eval_evasion(model.eval(), make_A_pert(A, edge_flip), X, y, test_idx))
            
    return accs


'''Below: parameters {dataset_name, attack_method, budget} are accessed from outer frame'''
def run_global_evasion_adaptive_exp(attack_configs, do_save_acc=True, do_save_flips=True, iter=200, init_model_name=None, **params):

    global_evasion_pgd_attack_fpath = path + f"exp/result/{args.data}/global_evasion_pgd_adaptive_{args.model}_{args.gamma}_{int(budget_ratio * 100)}_percent.yaml"

    for model_name, custom_model_params, custom_fit_params in attack_configs:
        print(f"Model:{model_name}")

        cur_params = params
        cleans, accs,  edge_flips, models = rep_global_evasion(
            args.data,
            *get_model(
                args.data, model_name, custom_model_params, custom_fit_params, as_paper=True
            ),
            attack_method=global_evasion_pgd,
            budget_ratio=budget_ratio, 
            return_model=True, 
            seed=0, 
            iter=iter, 
            init_As=load_rep_edge_flips(init_model_name, budget_ratio, attack_name, args.data) if init_model_name is not None else None, 
            **cur_params
        )
        print("Clean:",f"{np.mean(cleans)}±{np.std(cleans)}: {cleans}")
        print("Attacked:",f"{np.mean(accs)}±{np.std(accs)}: {accs}")
        if do_save_flips:
            save_rep_edge_flips(model_name, budget_ratio, attack_name=attack_name, flip_ls=edge_flips, dataset_name=args.data)
        # rep_save_model(model_name, budget_ratio, models)
        if do_save_acc:
            save_acc(cleans, accs, global_evasion_pgd_attack_fpath, model_name=model_name)


def run_global_evasion_transfer_exp(transfer_from_models, transfer_to_models, do_save_acc=True):

    rand_init=True # adaptive attack targets an architecture rather thana model that is specifically initialized

    global_evasion_pgd_transfer_attack_fpath = path + f'my_exp/{args.data}_acc_res/global_evasion_pgd_transfer.yaml'

    # transfer_to_models = [['rw_no_att_appnp_precond', {}, {}], ['rw_l21_appnp_precond', {}, {}]]
    
    # Masking
    # transfer_from_models = [
    #     ['gnn_guard_with_soft_mask', {}, {}], ['gnn_guard_with_hard_mask', {}, {}], ['gnn_guard', {}, {}], ['gnn_guard_with_mask', {}, {}], ['rw_soft_mcp_appnp_precond', {}, {}], ['rw_mcp_appnp_precond', {}, {}]
    # ]  
    # transfer_to_models = [
    #     ['gnn_guard_with_soft_mask', {}, {}], ['gnn_guard_with_hard_mask', {}, {}], ['gnn_guard', {}, {}], ['gnn_guard_with_mask', {}, {}], ['rw_soft_mcp_appnp_precond', {}, {}], ['rw_mcp_appnp_precond', {}, {}]
    # ]    


    for transfer_to_model, _, _ in transfer_to_models:

        for transfer_from_model, _, _ in transfer_from_models:
            try:
                model, _ = get_model(transfer_to_model)
                accs = rep_transfer_evasion(
                    args.data,
                    rep_load_model(transfer_to_model, 0 if rand_init else budget_ratio, model, dataset_name=args.data),
                    load_rep_edge_flips(transfer_from_model, budget_ratio, attack_name, dataset_name=args.data)
                )
                if len(accs) == 0:
                    raise ValueError
                print(f'Success: transfer from {transfer_from_model} to {transfer_to_model}')
                print(f'acc: {accs}')
                if do_save_acc:
                    save_acc(
                        accs,
                        global_evasion_pgd_transfer_attack_fpath,
                        lambda acc_dict, dict_args: {dict_args['trans_to']: {dict_args['trans_from']: {dict_args['budget']: acc_dict}}}, 
                        trans_to=transfer_to_model, 
                        trans_from=transfer_from_model, 
                        budget=f'{budget_ratio:.3f}'
                    )
            except Exception as e:

                print(f'During transfer from {transfer_from_model} to {transfer_to_model}, exception occurs.')
                print('Last exception message:', e)


if __name__ == '__main__':
    os.makedirs(path+f'log/{args.data}/attack', exist_ok=True)
    sys.stdout = open(path+f'log/{args.data}/attack/{args.model}_norm{args.norm}_gamma{args.gamma}.log', 'w', buffering=1)

    get_model = get_model_default
    attack_name = "global_evasion_PGD"

    for budget_ratio in [0.05, 0.1, 0.2, 0.3, 0.4]:
        print(f"Budget: {budget_ratio}")
        model_params = {'gamma': args.gamma, 'norm': args.norm}
        run_global_evasion_adaptive_exp([[args.model, model_params, {'max_epoch': 300}]])
    
    sys.stdout.close()