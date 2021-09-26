# hideandseek
deep learning and privacy preserving deep learning library.

Currently integrating from experiment codes. (26.9.2021.)

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

    model.to(device)
    node.step(local_T=20, horizon='epoch') # trains for 20 epochs
    # node.step(local_T=1000, horizon='step') # trains for 1000 steps
    model.cpu()

    node.save()

    test_results = hs.eval.test(node)
    scores = hs.eval.scores(test_results)
