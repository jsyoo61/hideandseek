# hideandseek
Highly modularized deep learning training library.

Why use `hideandseek`?

- Easy training & saving deep learning models along with other modules (ex: preprocessing modules) required in inference
- Run multiple deep learning experiments in parallel on multiples GPUs (powered by [hydra](https://hydra.cc/docs/intro/), and python multiprocessing)
- Design and analyze experiments scientifically by modifying variables (powered by [hydra](https://hydra.cc/docs/intro/))

- Modularized machine learning pipeline allows using the same script for all types of experiments
- The same training code can be run in privacy preserving setting by minimal modifications

Currently prettifying codes. (30.10.2022.)

    import torch
    import torch.nn as nn

    # Generate data
    x = torch.rand(200,1)
    y = 5*x+2

    network = nn.Linear(1,1)
    dataset = torch.utils.data.TensorDataset(x, y)
    criterion = nn.MSELoss()
    cfg = {
    'lr': 1e-2,
    'batch_size': 32,
    'epoch': 10 # optional
    }

    # Training configuration. All you need to train a neural network
    kwargs = {
    'network':network,
    'dataset':dataset,
    'cfg_train':cfg,
    'criterion':criterion,
    'name': 'Test' # optional
    }
    trainer = hs.N.Node(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.network.to(device)

    # Train for predefined number of epochs
    trainer.train() # Train for predefined number of epochs
    trainer.train(5) # Train for specified number of epochs
    trainer.train(epoch=5) # Same thing with trainer.train(5)
    trainer.train(step=500) # Train for specified number of updates

    trainer.network.cpu()

and simply run multiple batch of experiments with a single line command such as:

    python train.py -m lr=1e-3,1e-2 batch_size=32,64 "random_seed=range(0,5)" \
    hydra/launcher=joblib hydra.launcher.n_jobs=8
    # Runs total of 2*2*5=40 batch of experiments, with 8 processes at a time. Experiment results are stored in hydra.sweep.dir which can be overridden.

To do
- [ ] Draw figures to explain hideandseek
- [ ] GUI for generating experiment scripts when conducting variable sweeps
