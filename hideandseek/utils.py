from glob import glob
import logging
import os

from omegaconf import OmegaConf
import pandas as pd
import torch

import tools as T
import tools.torch

log = logging.getLogger(__name__)

# %%
'''
Should eliminate the following two.
Too inconsistent
'''
def exp_path(path=None, makedirs=True):
    path = '.' if path is None else path
    exp_path = T.Path(path)
    exp_path.network = 'network' # Save model while training
    exp_path.result = 'result' # Training results
    exp_path.model = 'model' # Save model. Package to be transferred
    if makedirs:
        exp_path.makedirs()
    return exp_path

def exp_setting(cfg, path=None):
    '''
    cfg must have the following structure:

    cfg:
      gpu_id: 0 # id of gpu, may be None
      random:
        seed: [int] # random seed
        strict: [bool] # strict random. If True, torch.backends.cudnn.benchmark=False and torch.backends.cudnn.deterministic=True
    '''
    if 'gpu_id' in cfg:
        gpu_id = cfg.gpu_id
    else:
        log.info('gpu_id not found in cfg')
        gpu_id = None

    # TODO: elastic adjustment of cfg.random

    # Print current experiment info
    log.info(OmegaConf.to_yaml(cfg))
    log.info(os.getcwd())

    # Set GPU for current experiment
    device = T.torch.multiprocessing_device(gpu_id) # To use distributed gpus, gpu_id must be None!
    log.info(device)

    T.random.seed(cfg.random.seed, strict=cfg.random.strict)
    path = exp_path(path)
    return device, path

def walk(walk_dir):
    '''
    Returns all directories with .hydra/ 
    '''
    dir_list_hydra = glob(os.path.join(walk_dir,'*/.hydra'))
    dir_list_hydra = list(map(os.path.dirname, dir_list_hydra))

    for subdir in T.os.listdir(walk_dir, join=True, isdir=True):
        if subdir not in dir_list_hydra:
            dir_list_hydra.extend(walk(subdir))

    return dir_list_hydra

def overrides_to_dict(listconfig):
    '''
    convert .hydra/overrides.yaml (hydra) into dict
    '''
    d = {}
    for cfg in listconfig:
        key, value = cfg.split('=')
        d[key]=value
    return d

def load_cfg(subdir):
    cfg = OmegaConf.load(os.path.join(subdir, '.hydra/config.yaml'))
    overrides = overrides_to_dict(OmegaConf.load(os.path.join(subdir, '.hydra/overrides.yaml')))
    return cfg, overrides

def load_cfg_sweep(l_subdir):
    '''
    Returns full set of experiment sweep configs
    '''
    cfg_overides = list(map(load_cfg, l_subdir))
    cfg_list, overrides_list = zip(*cfg_overides)
    df_cfg, df_overrides = pd.DataFrame(cfg_list), pd.DataFrame(overrides_list)
    independent_var = independent_var(df_overrides)

    return df_cfg, df_overrides

def independent_var(df_cfg):
    '''
    Returns the independent variable of a list of cfgs
    '''
    # df_cfg = pd.DataFrame(cfg_list)
    independent_variables = {}
    for column in df_cfg.columns:
        try: # Unhashable values raise Error
            cfg_list = df_cfg[column].unique().tolist()
        except:
            cfg_list = df_cfg[column].astype(str).unique().tolist()

        if len(cfg_list) > 1:
            independent_variables[column] = cfg_list
    
    return independent_variables

# %%
def model_type(model):
    '''
    Return library type of the given model.
    '''
    if issubclass(type(model), sklearn.BaseEstimator):
        return 'sklearn'
    elif issubclass(type(model), torch.nn.Module):
        return 'torch'
    else:
        return 'unknown'

def extract_dataset(dataset):
    if hasattr(dataset, 'get_x_all'):
        x = dataset.get_x_all()

    return {'x': x, 'y': y}

def add_batch_dim(data):
    """
    adds batch dimension (axis=0)

    Parameters
    ----------
    data : numpy.ndarray or dict of arrays

    Returns
    -------
    data : numpy.ndarray or dict of arrays with axis=0 dimension added
    """
    if isinstance(data, dict):
        data = {k: v[None,...] for k, v in data.items()}
    else:
        data = data[None,...]
    return data

def remove_batch_dim(data):
    if isinstance(data, dict):
        data = {k: v.squeeze(0) for k, v in data.items()}
    else:
        data = data.squeeze(0)
    return data

# %%
def reproducible_worker_dict():
    '''Generate separate random number generators for workers,
    so that the global random state is not consumed,
    thereby ensuring reproducibility'''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {'worker_init_fn': seed_worker, 'generator': g}
