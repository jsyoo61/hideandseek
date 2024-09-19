# hideandseek
Highly modularized deep learning training library.

Why use `hideandseek`?

- Easy training & saving deep learning models along with other modules (ex: preprocessing modules) required in inference
- Run multiple deep learning experiments in parallel on multiples GPUs (powered by [hydra](https://hydra.cc/docs/intro/), and python multiprocessing)
- Design and analyze experiments scientifically by modifying variables (powered by [hydra](https://hydra.cc/docs/intro/))

- Modularized machine learning pipeline allows using the same script for all types of experiments
- The same training code can be run in privacy preserving setting by minimal modifications

Look at `Simple/train.py` for simple run cases

Single run:
    
    python train.py lr=1e-3 batch_Size=32 random_seed=0

Multirun with batch of experiments (Hyperparameter sweep):

    python train.py -m lr=1e-3,1e-2 batch_size=32,64 "random_seed=range(0,5)" \
    hydra/launcher=joblib hydra.launcher.n_jobs=8
    # Runs total of 2*2*5=40 batch of experiments, with 8 processes at a time. Experiment results are stored in hydra.sweep.dir which can be overridden.

To do
- [ ] Draw figures to explain hideandseek
- [ ] GUI for generating experiment scripts when conducting variable sweeps
