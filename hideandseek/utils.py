import os
import logging

import torch

import tools as T
import tools.torch
from omegaconf import OmegaConf as OC

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
    log.info(OC.to_yaml(cfg))
    log.info(os.getcwd())

    # Set GPU for current experiment
    device = T.torch.multiprocessing_device(gpu_id) # To use distributed gpus, gpu_id must be None!
    log.info(device)

    T.random.seed(cfg.random.seed, strict=cfg.random.strict)
    path = exp_path(path)
    return device, path

# %%
def extract_dataset(dataset):
    if hasattr(dataset, 'get_x_all'):
        x = dataset.get_x_all()
    

    return {'x': x, 'y': y}


# %%
class Dataset(torch.utils.data.Dataset):
    '''
    inherit this class and update: __init__,
    '''
    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self):
            data = self[self._i]
            self._i +=1
            return data
        else:
            raise StopIteration

    def __getitem__(self, idx):
        x, y = self.get_x(idx), self.get_y(idx)

        return {'x': x, 'y': y}

    # def __len__(self):
    #     return len(self.y)

    def get_x(self, idx):
        # Should return x
        pass
    def get_y(self, idx):
        # Should return y
        pass

    def get_y_all(self):
        # Optional for fast computation when getting loss weights. return all y
        pass

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self, idx):
        return torch.as_tensor(self.x[idx])

    def get_y(self, idx):
        return torch.as_tensor(self.y[idx])

    def get_y_all(self):
        return torch.as_tensor(self.y)

    def __len__(self):
        return len(self.y)

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
