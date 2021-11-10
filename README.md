# hideandseek
privacy preserving deep learning library.

Why use `hideandseek`?

- Easy training & saving deep learning models along with other modules (ex: preprocessing modules) required in inference
- Run multiple deep learning experiments in parallel on multiples GPUs (powered by [hydra](https://hydra.cc/docs/intro/), and python multiprocessing)
- Design and analyze experiments scientifically by modifying variables (powered by [hydra](https://hydra.cc/docs/intro/))

- Modularized machine learning pipeline allows using the same script for all types of experiments
- The same training code can be run in privacy preserving setting by minimal modifications

Currently integrating from experiment codes. (30.10.2021.)

    import torch
    from omegaconf import OmegaConf
    import hideandseek as hs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load('config.yaml') # omegaconf.OmegaConf.DictConfig object
    model = DNN() # torch.nn.Module object
    train_dataset = dataset # torch.utils.data.Dataset object
    kwargs = {
      'model': model,
      'dataset': train_dataset,
      'cfg_train': cfg,
      'criterion': criterion,
    }
    node = hs.Node(**kwargs)

    node.model.to(device)
    node.step(local_T=20, horizon='epoch') # trains for 20 epochs
    # node.step(local_T=1000, horizon='step') # trains for 1000 steps
    node.model.cpu()

    node.save()

    test_results = hs.eval.test(node)
    scores = hs.eval.scores(test_results)

To do
- [ ] Migrate modules from experiment codes
- [ ] Draw figures to explain hideandseek
- [ ] GUI for generating experiment scripts when conducting variable sweeps
