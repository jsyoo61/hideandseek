# %%
import logging
import os
import shutil
from pathlib import Path

# %%
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd

import dataset as D
import hideandseek as hs
import tools as T

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(os.getcwd(),'conf'), job_name='train')
    overrides = []
    cfg = hydra.compose(config_name='train1', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    log.info=print

# %%
@hydra.main(config_path='conf', config_name='train', version_base='1.2')
def main(cfg: DictConfig) -> None:
    # %%
    # Print current experiment info
    log.info(OmegaConf.to_yaml(cfg))

    # Set GPU for current experiment if there's multiple gpu in the environment
    device = T.torch.multiprocessing_device(gpu_id=cfg.gpu_id)
    T.torch.seed(cfg.random.seed, strict=cfg.random.strict)
    log.info(f'device: {device}')

    # Assumes the process runs in a new directory (hydra.job.cwd==True)
    path_dict = {
        'network': Path('network'),
    }
    log.info(f'CWD: {os.getcwd()}')

    # %%
    # Load data
    ds_train, ds_val, ds_test = D.get_dataset(cfg)

    # %%
    # Neural network (Sometimes need to give data sample or data info to dynamically change network architecture)
    network = hydra.utils.instantiate(cfg.nn)
    network.to(device)

    # %%
    metrics = {
        'accuracy': hs.E.accuracy_score,
    }

    kwargs = {
        'network': network,
        'train_dataset': ds_train,
        'cfg_train': OmegaConf.to_container(cfg.train['update'], resolve=True),
        'criterion': hydra.utils.instantiate(cfg.train.criterion, dataset=ds_train),
        'network_dir': path_dict['network'],

        'cfg_val': OmegaConf.to_container(cfg.train.validation, resolve=True),
        'val_dataset': ds_val,
        'val_metrics': metrics,
        # 'verbose': True,
        'verbose': False,
        'amp': cfg.amp
    }

    # %%
    trainer = hs.Trainer(**kwargs)

    # %%
    trainer.train()  
    trainer.load_best_model()

    # %%
    # Testing
    trainer.network.to(device)
    result = hs.E.test_model(trainer.model, ds_test, batch_size=cfg.train.validation.batch_size)
    scores = hs.E.evaluate(results=result, metrics=hs.E.classification_report_full)

    scores = hs.E.classification_report_full(result, ovr=True)
    log.info(f'Evaluation scores: {pd.DataFrame(scores).T}')

    T.save_pickle(scores, 'scores.p')

    # %%
    if cfg.save_model:
        trainer.save('network')
    else:
        shutil.rmtree('network')
    if os.path.exists('network_temp'): shutil.rmtree('network_temp')

    if cfg.save_result:
        T.save_pickle(result, 'result.p')

# %%
if __name__ == '__main__':
    main()
