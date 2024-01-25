import os
import logging

from omegaconf import OmegaConf
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
    exp_path.MODEL = 'model' # Save model while training
    exp_path.RESULT = 'result' # Training results
    exp_path.NODE = 'node' # Save node info & model. Package to be communicated
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
