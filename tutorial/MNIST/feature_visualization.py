# %%
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
import hideandseek as hs

import matplotlib.pyplot as plt
import numpy as np

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
class Crit(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network=network

    def forward(self, x):
        return self.network(x)[:,0]

objective = Crit(model)

# %%

maximize=True
lr=1e-1
updates=1000
jitter_ratio=0.05
threshold=200
verbose=True

data= torch.randn(3, 3, 224, 224).to(device)
# preprocess=models.ResNet18_Weights.IMAGENET1K_V1.transforms()
preprocess=nn.Sigmoid()
regularization = transforms.RandomCrop(data.shape[-2:], padding=(int(jitter_ratio*data.shape[-2]),int(jitter_ratio*data.shape[-1])), pad_if_needed=True) # Random Jitter

# %%
features = hs.fv.feature_visualize(objective, data, maximize, lr, updates, threshold, preprocess, regularization, verbose=verbose)

# %%
images=np.transpose(features.cpu().numpy(), (0, 2, 3, 1))
plt.imshow(images[0])
plt.imshow(images[1])
plt.imshow(images[2])