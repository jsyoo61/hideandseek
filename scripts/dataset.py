

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

# %%
def MNIST(root):
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())

    x, y = train_dataset[0]
    input_shape = x.shape
    train_data = train_dataset.data.unsqueeze(1)/255
    test_data = test_dataset.data.unsqueeze(1)/255

    dataset_kwargs = {'get_f': None, 'grayscale': True, 'targets_type': 'category'}
    train_dataset = ImageDataset(data=train_data, targets=np.array(train_dataset.targets), **dataset_kwargs)
    test_dataset = ImageDataset(data=test_data, targets=np.array(test_dataset.targets), **dataset_kwargs)

    data = {
    'info': {
        'n_classes': 10,
        'in_channels': 1,
        'input_shape': input_shape
        },
    'dataset': {
        'train': train_dataset,
        'test': test_dataset
        },
    }
    return data
