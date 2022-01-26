'''join .csv or .xlsx files
result.p must exist in each experiment folder'''
import argparse
import itertools as it
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import tools as T

parser = argparse.ArgumentParser(description='description')
parser.add_argument('--exp_dir', type=str, help='Experiment directory')
parser.add_argument('--result', type=str, default='result/result.p', help='experiment result to group')
parser.add_argument('--savename', type=str, default=None, help='')
args = parser.parse_args()

# %%
def exp_name(exp_dict, name):
    '''
    If "exp" in exp_dict, then add {"exp`":name}.
    '''
    exp_key = 'exp'
    exp_key_ = exp_key
    i=1
    # import pdb; pdb.set_trace()
    while exp_key_ in exp_dict.keys():
        exp_key_ = exp_key+str(i)
        i+=1
    # if i>=3:
    #     import pdb; pdb.set_trace()
    exp_dict[exp_key_] = name
    return exp_dict

def walk(walk_dir, exp_dict={}):
    print(walk_dir, end='')
    exp_dict = exp_dict.copy()

    if len(T.os.listdir(walk_dir, isdir=True)) == 0:
        print('Why am I in an empty directory?')
    else:
        print('')
    for subdir in T.os.listdir(walk_dir, join=True, isdir=True):
        if os.path.exists(os.path.join(subdir, '.hydra')):
            join_result(subdir, exp_dict)
        else:
            exp_dict_subdir = exp_dict.copy()
            exp_dict_subdir = exp_name(exp_dict_subdir, os.path.basename(subdir))
            # import pdb; pdb.set_trace()
            walk(subdir, exp_dict=exp_dict_subdir)

def elastic_config(subdir):
    if os.path.exists(os.path.join(subdir, '.hydra/config.yaml')):
        return os.path.join(subdir, '.hydra/config.yaml')
    elif os.path.exists(os.path.join(subdir, '.hydra/exp_config.yaml')):
        return os.path.join(subdir, '.hydra/exp_config.yaml')
    else:
        raise Exception(f'no proper config found: {os.listdir(subdir)}')

def join_result(subdir, exp_dict={}):
    cfg = OmegaConf.load(elastic_config(subdir))
    cfg = OmegaConf.to_container(cfg, resolve=False)

    result_path = os.path.join(subdir, args.result)
    if os.path.exists(result_path):
        result = T.load_pickle(result_path)
        result.update(cfg)
        result.update(exp_dict)
        result = T.unnest_dict(result)
        result['subdir']=subdir
        result_list.append(result)
        print(f'[exp: {subdir}]')
    else:
        print(f'[exp: {subdir}] result not found')
        result = T.unnest_dict(cfg)
        result['subdir']=subdir
        result_list.append(result)

args.savename = args.savename if args.savename is not None else os.path.splitext(os.path.basename(args.result))[0]

result_list = []
# Retrieve directories to analyze
walk(args.exp_dir, exp_dict={'exp':os.path.basename(args.exp_dir)})
# Load all scores 
result_table = pd.DataFrame(result_list)
result_table.to_excel(os.path.join(args.exp_dir, f'{args.savename}.xlsx'), index=False)
