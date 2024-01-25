# %%
import logging

import hydra
from omegaconf import OmegaConf, DictConfig

import torch

# %%
import sys
import os
print(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
import hideandseek as hs

PROJECT_DIR=os.path.join(os.getcwd())

# import os
# os.getcwd()
# PROJECT_DIR='/home/jaesungyoo/programming/hideandseek/tutorial'
# os.chdir(PROJECT_DIR)
# os.listdir()

# %%
# to enable logging from hideandseek
# When using command line interface, there's no need to set the log level (feature from hydra)
log = logging.getLogger()
log.setLevel(logging.INFO)

# %%
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'))
overrides = [
'data=MNIST',
'model=Resnet',
'train.batch_size=128',
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
log.info(data.keys())

criterion = hydra.utils.instantiate(cfg.train.loss, dataset=train_dataset)

model = hydra.utils.instantiate(cfg.model, info=data['info'])
log.info(model)
kwargs = {'model': model, 'dataset': train_dataset, 'validation': None, 'cfg_train': cfg.train,
        'criterion': criterion, 'model_dir': path['model'], 'node_dir': path['node'], 'verbose': True, 'amp': False}

node = hs.N.Node(**kwargs)

# %%
# train
node.model.to(device)
node.train() # train for specified epoch in cfg_train

# %%
node.train(epoch=2) # train for explicit number of epochs
node.train(step=10) # train for explicit number of updates
node.model.cpu()

# %%
# Save best model (Early stopping)
node.save(best=True)
node.load()

# %%
# Evaluation
node.model.to(device)
results = hs.E.test(node, test_dataset, test_batch_size=256)
log.info(result.keys())

score = hs.E.classification_score(results)
