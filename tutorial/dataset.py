import torchvision
import torchvision.transforms as transforms

def MNIST(root):
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transforms=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transforms=transforms.ToTensor())

    data = {
    'info': {
        'n_classes': 10,
        'in_channel': 1,
        },
    'dataset': {
        'train': train_dataset,
        'test': test_dataset
        }
    }
    return data

def CIFAR10(root):
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transforms=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transforms=transforms.ToTensor())

    data = {
    'info': {
        'n_classes': 10,
        'in_channel': 3,
        },
    'dataset': {
        'train': train_dataset,
        'test': test_dataset
        }
    }
    return data
