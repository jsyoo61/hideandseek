import logging

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from sklearn.preprocessing import LabelEncoder

import tools as T
import hideandseek as hs

log = logging.getLogger(__name__)

# %%
'''
get_dataset() returns train, validation (optional), test dataset objects
CustomDataset()
'''

def get_dataset(cfg):
    # Load signal cohort
    data = T.load_pickle(cfg.dir.data)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(data['label'])
    labels = y_encoder.classes_
    
    # Split data
    train_i, val_i, test_i = hydra.utils.instantiate(cfg.cv, y=y) # get indices based on y value (Could use Stratified split)
    log.info(f'train/val/test: {len(train_i)/len(y)}/{len(val_i)/len(y)}/{len(test_i)/len(y)}')    
    T.save_pickle(train_i, 'train_i.p'), T.save_pickle(val_i, 'val_i.p'), T.save_pickle(test_i, 'test_i.p')
    train_data, val_data, test_data = data[train_i], data[val_i], data[test_i]

    # Make torch dataset
    ds_train, ds_val, ds_test = CustomDataset(train_data), CustomDataset(val_data), CustomDataset(test_data)

    return ds_train, ds_val, ds_test

class CustomDataset(hs.D.Dataset):
    targets_type = 'categorical'
    def __init__(self, data):
        self.x_i = data['x']
        self.y = data['y']

    def __len__(self):
        return len(self.y)

    def get_x(self, idx):
        signal = T.load_pickle(self.x_i[idx])
        signal = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0) # Add channel dimension
        return signal 

    def get_y(self, idx):
        return self.y[idx]
    
    def get_y_all(self):
        return self.y
    