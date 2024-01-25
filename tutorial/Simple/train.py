# %%
import torch
import torch.nn as nn

# Generate data
x = torch.rand(200,1)
y = 5*x+2

model = nn.Linear(1,1)
dataset = torch.utils.data.TensorDataset(x, y)
criterion = nn.MSELoss()
cfg = {
'lr': 1e-2,
'batch_size': 32,
'epoch': 10 # optional
}

# Training configuration. All you need to train a neural network
kwargs = {
'model':model,
'dataset':dataset,
'cfg_train':cfg,
'criterion':criterion,
'name': 'Test' # optional
}
trainer = hs.N.Node(**kwargs)

# Train for predefined number of epochs
trainer.train() # Train for predefined number of epochs
trainer.train(5) # Train for specified number of epochs
trainer.train(epoch=5) # Same thing with trainer.train(5)
trainer.train(step=500) # Train for specified number of updates
