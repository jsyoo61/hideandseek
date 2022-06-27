import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Load Dataset
# Load torch dataset that can be fed into torch.utils.data.DataLoader directly
train_dataset = datasets.FashionMNIST('data/', train=True, download=True)
test_dataset = datasets.FashionMNIST('data/', train=False, download=True)
x, y = train_dataset[0] # x: PIL image (search google for more info.)
x # Show image
x=np.array(x) # Convert PIL image to numpy
x.shape

# %% Load Dataset3
# To load dataset as tensors, not PIL image
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST('data/', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('data/', train=False, download=True, transform=transform)
x, y = train_dataset[0]
plt.matshow(x[0], cmap=matplotlib.cm.binary)

# %%
from sklearn.model_selection import ParameterGrid, GridSearchCV
n_input = 28*28
n_hidden_list = [300, 200, 100, 10]
activation_list = [nn.ReLU()]*4
model = DNN(n_input, n_hidden_list, activation_list)
model

param_grid = [
{'n_hidden_list' : [[200,100,10]],
'activation_list' : [[nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU()]],
'lr' : [1e-2, 1e-3, 1e-4],
'batch_size': [32, 64],},

{'n_hidden_list' : [[300,200,100,10]],
'activation_list' : [nn.Sigmoid(), nn.ReLU(), [nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(), nn.LeakyReLU()]],
'lr' : [1e-2, 1e-3, 1e-4],
'batch_size': [32, 64],}
]

param_grids = list(ParameterGrid(param_grid))
param_grids

# %% Dataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)

# %% Hyperparameters
n_epoch = 3
lr = 1e-3
batch_size = 64
# validation_interval =

optimizer = optim.Adam(model.parameters(), lr=lr)

# %% Training
# batch_size
# for epoch in range(1, n_epoch+1):
#     for i in range(len(train_imgs)//batch_size+1):
#         start = i*batch_size
#         end = (i+1)*batch_size
#         x = train_imgs[start:end]

train_dataset = MyDataset(train_imgs, train_label)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
dev_dataset = MyDataset(dev_imgs, dev_label)
dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()

# %% Tracking loss for each epoch
# 1. average loss list
train_loss_list = []
train_loss_iter_list = []
# epoch 돌리고
train_loss_list.append(np.average(train_loss_iter_list))

# 2. class averagemeter
class AverageMeter(object):
    """Computes and stores the average and current value
    val
    avg
    sum
    count
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def step(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

train_loss_meter = AverageMeter() # Loss of single epoch
dev_loss_meter = AverageMeter()
train_loss_list = [] # Loss over each epoch
dev_loss_list = []

# %% Training
for epoch in range(1, n_epoch+1):
    train_loss_meter.reset()
    dev_loss_meter.reset()
    for i, (x, y) in enumerate(train_dataloader):
        y_hat = model(x)
        loss = criterion(y_hat, y)

        optimizer.zero_grad() # Erase gradient history
        loss.backward() # differentiate
        optimizer.step() # apply

        train_loss_meter.step(loss.item())
        print('[Epoch: %s/%s][Batch: %s/%s][Loss: %.5f (%.5f)]'%(epoch, n_epoch, i, len(train_dataloader), loss.item(), train_loss_meter.avg))

        # Validation

        # 1. Check Iteration
        # if i % validation_interval == 0:
        #     with torch.no_grad():
        #         for x, y in dev_dataloader:
        #             y_hat = model(x)
        #             loss = criterion(y_hat, y)
        #
        #             dev_loss_list.append(loss.item())
    train_loss_list.append(train_loss_meter.avg)
    # Validation
    # 2. Check epoch
    # (Validation every epoch)
    with torch.no_grad():
        for x, y in dev_dataloader:
            y_hat = model(x)
            loss = criterion(y_hat, y)

            dev_loss_meter.step(loss.item())
    print('Validation Loss: %.5f'%(dev_loss_meter.avg))
    dev_loss_list.append(dev_loss_meter.avg)


# %% Ploss loss curve
fig = plt.figure()
plt.plot(train_loss_list)
plt.plot(dev_loss_list)
plt.legend(['train_loss', 'dev_loss'])
plt.show()

# %% Testing
# batch normalization layers
# training 할 때랑, testing 할 때랑 연산방법이 다름
model.eval()

test_dataset = MyDataset(test_imgs, test_label)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

y_hat_list = []
for x, y in test_dataloader:
    with torch.no_grad():
        # Accuracy check
        y_hat = model(x)
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_list.append(y_hat)

        # 변환모듈이고, 변환된 이미지/음성/데이터 저장하고싶으면 여기서 저장
y_hat_list = torch.cat(y_hat_list, dim=0)
y_hat_list.shape
y_pred = torch.argmax(y_hat_list, dim=1).numpy()

# %% Accuracy
from sklearn.metrics import accuracy_score
accuracy = np.sum(y_pred == test_label) / len(test_dataset)
print(accuracy)
accuracy = accuracy_score(test_label, y_pred)
print(accuracy)

# Iterating over array to compute accuracy
accuracy = 0.
for y_t, y_p in zip(test_label, y_pred):
    accuracy += y_t == y_p
accuracy = accuracy / len(test_dataset)
print(accuracy)
y_pred = torch.as_tensor(y_pred)

# %% Wrapping into function
from sklearn.metrics import accuracy_score
def test(model, imgs, label):
    dataset = MyDataset(imgs, label)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    y_hat_list = []
    for x, y in dataloader:
        with torch.no_grad():
            # Accuracy check
            y_hat = model(x)
            y_hat = torch.softmax(y_hat, dim=1)
            y_hat_list.append(y_hat)

            # 변환모듈이고, 변환된 이미지/음성/데이터 저장하고싶으면 여기서 저장
    y_hat_list = torch.cat(y_hat_list, dim=0)
    y_hat_list.shape
    y_pred = torch.argmax(y_hat_list, dim=1).numpy()

    # Accuracy
    accuracy = accuracy_score(label, y_pred)
    return accuracy

train_accuracy = test(model, train_imgs, train_label)
dev_accuracy = test(model, dev_imgs, dev_label)
test_accuracy = test(model, test_imgs, test_label)

# %% Dimension check
x, y = next(iter(test_dataloader))
# (Batch dimension, necessary dimensions ...)
x.shape # (batch, 28, 28)
y.shape # (batch,)

y_hat # (batch, 10) -> dimension number: (0, 1)
torch.softmax(y_hat, dim=1) # Softmax는 10개에 대해서 해줘야 돼서 dim=1

y_hat_list # [(batch, 10), (batch, 10), ...] --> (number of batch, batch, 10)
y_hat_list = torch.cat(y_hat_list, dim=0)
# Concatenate erases the selected dimension
# (batch, 10) -- dim=0 --> (number of batch * batch, 10)
# (batch, 10) -- dim=1 --> (batch, 10 * number of batch)
# np.hstack, np.vstack == np.concatenate(dim=1), np.concatenate(dim=0)

# np.concatenate vs np.stack
# concatenate: (batch, 10) -- dim=0 --> (number of batch * batch, 10)
# stack: (batch, 10) -- dim=0 --> (number of batch, batch, 10) # Add new dimension
x= [np.arange(30).reshape(3,10) for i in range(5)]
np.concatenate(x, axis = 0).shape
np.stack(x, axis=0).shape

# %% Confusion matrix
# scikitlearn confusion matrix
'''
Area Under ROC curve (AUROC)
(ROC curve: Reciever Operating Curve)
confusion matrix
'''



# %% Save & Load
model.parameters() # gradient descent로 학습하는 파라미터 == weight, bias.
model.state_dict() # 모델에 존재하는 모든 파라미터 == weight, bias, (mean, var (batch norm)), etc.
torch.save(model.state_dict(), 'model.pt')

state_dict = torch.load('model.pt')
model == DNN() # 미리 만들어 놓고
model.load_state_dict(state_dict) # 모델 load
