import os
import logging

import tools as T
import tools.torch
from omegaconf import OmegaConf as OC

log = logging.getLogger(__name__)

def exp_path(path=None, makedirs=True):
    path = '.' if path is None else path
    exp_path = T.Path(path)
    exp_path.MODEL = 'model' # Save model while training
    exp_path.RESULT = 'result' # Training results
    exp_path.NODE = 'node' # Save node info & model. Package to be communicated
    if makedirs:
        exp_path.makedirs()
    return exp_path

# %%
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
