import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

import tools as T

log = logging.getLogger(__name__)

def exp_setting(cfg):
    # Print current experiment info
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f'CWD: {os.getcwd()}')

    # Set GPU for current experiment
    device = T.torch.multiprocessing_device(gpu_id=cfg.gpu_id)
    T.torch.seed(cfg.random_seed, strict=cfg.random_strict)

    # Assumes the process runs in a new directory (hydra.cwd==True)
    path_dict = {
        'result': Path('result'),
        'model': Path('model'),
    }

    log.info(f'device: {device}')
    return device, path