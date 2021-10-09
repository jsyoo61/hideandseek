# %%
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

import hideandseek as hs

log = logging.getLogger()
log.setLevel(logging.INFO) # to enable logging from hideandseek

# %%
PROJECT_DIR=''

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'))
overrides = [
'data=MNIST',
'model=Resnet',
]
cfg = hydra.compose(config_name='train_cfg', overrides=overrides)
print(OmegaConf.to_yaml(cfg))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = {}

# %%
# Settings for training
data = hydra.utils.instantiate(cfg.data.dataset)
train_dataset, test_dataset = data['dataset']['train'], data['dataset']['test']

criterion = hydra.utils.instantiate(cfg.train.loss, dataset=train_dataset)

model = hydra.utils.instantiate(cfg.model, info=data['info'])
log.info(model)
kwargs = {'model': model, 'dataset': train_dataset, 'cv': None, 'cfg_train': cfg.train,
        'criterion': criterion, 'MODEL_DIR': path.MODEL, 'NODE_DIR': path.NODE.join('server'), 'name': 'server'}
node = hs.N.Node(**kwargs)

# %%
# train
node.model.to(device)
node.step(cfg.train.epoch)
node.model.cpu()

# %%
# Save best model (Early stopping)
node.save(best=True)
node.load()

# %%
node.model.to(device)
