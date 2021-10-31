# %% codecell
import os
import logging

import hydra
from omegaconf import OmegaConf as OC
from omegaconf import DictConfig

import hideandseek as hs

# %%
log = logging.getLogger(__name__)

# # %%
# if False:
#     # %%
#     # Load config
#     os.getcwd()
#     # PROJECT_DIR = '/zdisk/jaesungyoo/spatial_gene'
#     PROJECT_DIR = '/home/jaesungyoo/spatial_gene'
#     os.chdir(PROJECT_DIR)
#     os.listdir()
#
#     # Dummy class for debugging
#     class Dummy():
#         """ Dummy class for debugging """
#         def __init__(self):
#             pass
#     log = Dummy()
#     log.info=print
#
#     # %%
#     hydra.core.global_hydra.GlobalHydra.instance().clear()
#     hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='debug')
#     overrides = []
#     overrides = ['save_x=True']
#     overrides = ['criterion=bce_loss']
#     overrides = ['criterion=bce_loss', 'train.epoch=0']
#     cfg = hydra.compose(config_name='autoencoder', overrides=overrides)
#     print(OC.to_yaml(cfg))

# %%
@hydra.main(config_path='conf', config_name='train_cfg')
def main(cfg: DictConfig) -> None:
    # %%
    device, path = hs.U.exp_setting(cfg)

    # Load data
    data = hydra.utils.instantiate(cfg.data.dataset)
    dataset_train, dataset_test = data['dataset']['train'], data['dataset']['test']

    # criterion
    criterion = hydra.utils.instantiate(cfg.criterion)
    log.info(f'criterion: {criterion}')

    # %%
    # load model
    model = hydra.utils.instantiate(cfg.model,  data['info'])
    log.info(model)

    kwargs = {'model': model, 'dataset': dataset_train, 'cv': None, 'cfg_train': cfg.train,
            'criterion': criterion, 'MODEL_DIR': path.MODEL, 'NODE_DIR': path.NODE.join('default'), 'name': 'server', 'verbose': False, 'amp': cfg.amp}
    node = hs.N.Node(**kwargs)

    # %%
    node.model.to(device)
    node.step(no_val=True)
    # node.save(path.Node)

    # %%
    # Evaluate
    node.model.to(device)
    result = hydra.utils.instantiate(cfg.eval, model, data)
    node.model.cpu()
    score = hydra.utils.instantiate(cfg.scorer, result=result, data=data, path=path.RESULT)
    hs.E.save_score(score, path.RESULT)

    # %%

if __name__=='__main__':
    main()
