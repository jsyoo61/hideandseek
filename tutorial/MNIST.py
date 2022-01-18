# %%
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import hideandseek as hs

# %%
# to enable logging from hideandseek
# When using command line interface, there's no need to set the log level (feature from hydra)
log = logging.getLogger()
log.setLevel(logging.INFO)

# %%
import os
os.getcwd()
PROJECT_DIR='/home/jaesungyoo/programming/hideandseek/tutorial'
os.chdir(PROJECT_DIR)
os.listdir()

# %%
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'))
overrides = [
'data=MNIST',
'model=Resnet',
'train.batch_size=2048',
'train.epoch=10',
]
cfg = hydra.compose(config_name='train_cfg', overrides=overrides)
print(OmegaConf.to_yaml(cfg))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = {'model': 'exp/model', 'node': 'exp/node'}

# %%
# Settings for training
data = hydra.utils.instantiate(cfg.data.dataset)
train_dataset, test_dataset = data['dataset']['train'], data['dataset']['test']

criterion = hydra.utils.instantiate(cfg.train.loss, dataset=train_dataset)

model = hydra.utils.instantiate(cfg.model, info=data['info'])
log.info(model)
kwargs = {'model': model, 'dataset': train_dataset, 'cv': None, 'cfg_train': cfg.train,
        'criterion': criterion, 'MODEL_DIR': path['model'], 'NODE_DIR': path['node'], 'verbose': True, 'amp': True}

node = hs.N.Node(**kwargs)

# %%
# train
node.model.to(device)
node.step()
node.step(T=10)
node.model.cpu()

# %%
# Save best model (Early stopping)
node.save(best=True)
node.load()

# %%
# Evaluation
node.model.to(device)
hs.E.test(node, test_dataset)
