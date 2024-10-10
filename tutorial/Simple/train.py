# %%
import torch
import torch.nn as nn
import hideandseek as hs

# Creating synthetic data for model
x = torch.rand(200, 1)  # Generate 200 random samples (input features)
y = 5 * x + 2           # Define the target outputs using a simple linear equation

# Sets up a linear neural network model with has 1 input and 1 output
network = nn.Linear(1, 1)

# Creates dataset from the input and output data
dataset = torch.utils.data.TensorDataset(x, y)

# Choose loss function for model (Mean Squared Error Loss)
criterion = nn.MSELoss()

# Configuration dictionary for our training setup
cfg = {
    'lr': 1e-2,           # Learning rate for the optimizer
    'batch_size': 32,     # Batch size used for training
    'epoch': 10           # Number of epochs (optional, default training cycles)
}

# Prepare the keyword arguments for the trainer class (includes model, dataset, configuration, and loss function)
kwargs = {
    'network': network,        # Untrained neural network model
    'train_dataset': dataset,  # Dataset we are training
    'cfg_train': cfg,          # Configurations for training
    'criterion': criterion,    # Loss function for neural network
    'name': 'Test'             # Optional name for the training session
}

# Instantiate HideAndSeek with arguments
trainer = hs.Trainer(**kwargs)

# Train using the default epoch count
trainer.train() 

# Train for an additional 5 epochs
trainer.train(5)  

# Another way to train for 5 epochs, specifying it explicitly with the 'epoch' parameter
trainer.train(epoch=5)  

# Train for 500 steps/updates instead of by epoch count
trainer.train(step=500) 

# If needed, move the trained model back to CPU (useful if training on a GPU)
# trainer.network.cpu()