import numpy as np

import torch
import torch.utils.data as D
import torchvision
import torchvision.transforms as transforms

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

def CIFAR10(root):
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transforms=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transforms=transforms.ToTensor())

    data = {
    'info': {
        'n_classes': 10,
        'in_channels': 3,
        },
    'dataset': {
        'train': train_dataset,
        'test': test_dataset
        }
    }
    return data

# %%
def infer_targets_type(targets):
    classes = sorted(list(set(np.array(targets))))
    if set(classes)==set([0,1]):
        return 'binary'
    else:
        return 'category'

def infer_targets_dtype(targets_type):
    if targets_type == 'binary':
        return torch.float32
    elif targets_type == 'cateogry':
        return torch.long

class ImageDataset(D.Dataset):
    '''
    must specify get_f if you want to load directly from drive.
    '''
    def __init__(self, data, targets, get_f=None, grayscale=False, targets_type=None, targets_dtype=None):
        self.data = data
        self.targets = targets
        self.get_f = get_f
        self.grayscale = grayscale

        if targets_type is None:
            self._targets_type = infer_targets_type(self.targets)
        else:
            self._targets_type = targets_type
        if targets_dtype is None:
            self._targets_dtype = infer_targets_dtype(self._targets_type)
        else:
            self._targets_dtype = targets_dtype

    def __getitem__(self, idx):
        # Resize to 224?
        x = self.get_f(self.data[idx]) if self.get_f is not None else self.data[idx] # if get_f is none, it will most likely raise an error
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(self.targets[idx], dtype=self._targets_dtype)
        return {'x':x, 'y':y}

    def __len__(self):
        return len(self.targets)
