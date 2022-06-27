import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Model

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# 1. Load data


# 2. Hyperparameters
lr = 1e-4
batch_size = 150
n_epoch = 30

# 3. Create model
# model = Model(input_size=8, output_size=3, h_list=h_list)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
# lr_scheduler = optim.lr_scheduler()

train_dataset = DataLoader(x_train, batch_size=batch_size, shuffle=True) # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
criterion = nn.MSELoss()

# 4. SummaryWriter
writer = SummaryWriter()
train_meter = AverageMeter()
train_loss_list = []

# 4. Train
for epoch in range(1, n_epoch+1):
    train_meter.reset()
    for i, (x, y) in zip(count(1), train_dataset):
        N = len(x)

        x = x.cuda()
        y = y.cuda()

        y_hat = model(x)
        loss = criterion(y_hat, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss.item())

        train_meter.step(loss.item(), n=N)
        writer.add_scalar('Loss/train/iter', loss.item(), i)
        # 4-1. Tensorboard

        # 4-2. Print
        print('[Epoch: %s/%s][Iter: %s/%s][Loss: %s(%s)]'%(epoch, n_epoch, i, len(train_dataset), loss))

        # 4-3. Validation

    train_loss_list.append(train_meter.avg)
    writer.add_scalar('Loss/train/epoch', train_meter.avg, epoch)
    # Early stopping

    # Save intermediate model
    # Save only if validation loss has decreased

# 5. Save model
torch.save('model.pth', model.state_dict())
