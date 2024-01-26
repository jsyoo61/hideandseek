import os
import functools
import logging
import shutil

import hydra
from omegaconf import OmegaConf, DictConfig

# %%
import run # main run.py to execute
import tools as T

# %%
log = logging.getLogger(__name__)

# %%
def setup(cfg):
    log.info(f'Resuming experiment from dir: {cfg.dir} (cwd: {os.getcwd()})')

    # Load config
    cfg_resume = OmegaConf.load('.hydra/config.yaml')

    # Remove remnants
    l_dir = T.os.listdir(isdir=True)
    l_dir.remove('.hydra')
    for directory in l_dir:
        shutil.rmtree(directory)

    return cfg_resume

# %%
@hydra.main(config_path='config', config_name='resume_cfg', version_base='1.1')
def main(cfg: DictConfig) -> None: # Load configs automatically
    cfg_resume = setup(cfg)
    run.main.__wrapped__(cfg_resume) # Unwrapp hydra.main() which loads configs from the path set in hydraconfig==resume_cfg

# %%
if __name__=='__main__':
    main()
