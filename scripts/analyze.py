# %%
from glob import glob
import logging
import os
from pathlib import Path
import warnings

import hydra
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig

import tools as T

log = logging.getLogger(__name__)

# %%
if False:
    # %%
    if os.path.basename(os.getcwd())!='code':
        os.chdir('..')
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(os.getcwd(), 'config'), job_name='debug')
    overrides = ['dir.sweep=exp/sweep4']
    cfg = hydra.compose(config_name='analyze', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
# %%

def walk(walk_dir):
    '''
    Returns all directories with .hydra/ 
    '''
    dir_list_hydra = glob(os.path.join(walk_dir,'*/.hydra'))
    dir_list_hydra = list(map(os.path.dirname, dir_list_hydra))

    for subdir in T.os.listdir(walk_dir, join=True, isdir=True):
        if subdir not in dir_list_hydra:
            dir_list_hydra.extend(walk(subdir))

    return dir_list_hydra

def overrides_to_dict(listconfig):
    d = {}
    for cfg in listconfig:
        key, value = cfg.split('=')
        d[key]=value
    return d

def load_cfg(subdir):
    cfg = OmegaConf.load(os.path.join(subdir, '.hydra/config.yaml'))
    overrides = overrides_to_dict(OmegaConf.load(os.path.join(subdir, '.hydra/overrides.yaml')))
    return cfg, overrides

def load_cfg_sweep(l_subdir):
    '''
    Returns full set of experiment sweep configs
    '''
    cfg_overides = list(map(load_cfg, l_subdir))
    cfg_list, overrides_list = zip(*cfg_overides)
    df_cfg, df_overrides = pd.DataFrame(cfg_list), pd.DataFrame(overrides_list)
    var_ind = unique_var(df_overrides)

    return df_cfg, df_overrides, var_ind

def unique_var(df_cfg):
    '''
    Returns the independent variable of a list of cfgs
    '''
    # df_cfg = pd.DataFrame(cfg_list)
    independent_variables = {}
    for column in df_cfg.columns:
        try: # Unhashable values raise Error
            cfg_list = df_cfg[column].unique().tolist()
        except:
            cfg_list = df_cfg[column].astype(str).unique().tolist()

        if len(cfg_list) > 1:
            independent_variables[column] = cfg_list
    
    return independent_variables

# %%
def load_result(subdir):
    if os.path.exists(os.path.join(subdir, 'scores.p')):
        scores = T.load_pickle(os.path.join(subdir, 'scores.p'))
    else:
        warnings.warn(f'No scores found in {subdir}')
        scores = {}
    return scores

def load_exp_result(l_subdir):
    '''
    Returns full set of experiment sweep
    '''
    l_scores = list(map(load_result, l_subdir))
    df_scores = pd.DataFrame(l_scores)

    df_cfg, df_overrides, var_ind = load_cfg_sweep(l_subdir)

    return df_scores, df_overrides, df_cfg, var_ind

ignore_columns = ['random.seed', 'cv.split_i', 'cv.split_j']

def plot(df, target, groupby):
    fig, ax = plt.subplots()
    df.boxplot(column=target, by=groupby, grid=False, ax=ax)
    fig.tight_layout()
    fig.suptitle('')
    ax.xaxis.set_tick_params(rotation=45)
    fig.set_tight_layout(True)

    return fig, ax



# %%
@hydra.main(config_path='config', config_name='analyze', version_base='1.2')
def main(cfg: DictConfig) -> None:

    # %%
    print(OmegaConf.to_yaml(cfg))
    path_save = Path(cfg.dir.save)/cfg.dir.sweep.split(os.path.sep)[-1]
    path_save.mkdir(parents=True, exist_ok=True)
    
    # %%
    l_subdir = walk(cfg.dir.sweep)
    df_scores, df_overrides, df_cfg, var_ind = load_exp_result(l_subdir)

    var_ind_real = var_ind.copy()
    for column in ignore_columns:
        if column in var_ind_real:
            var_ind_real.pop(column)
    var_ind_real = list(var_ind_real.keys())

    # %%
    df = pd.concat([df_scores, df_overrides], axis=1)
    df.dropna(subset=['l1','mse', 'r2'], inplace=True)
    print(df.shape)
    d_dtypes = {
    'train.update.lr':'float64',
    'train.update.batch_size':'int64',
    'random.seed': 'int64',
    'cv.n_splits': 'int64',
    'cv.m_splits': 'int64',
    'cv.split_i': 'int64',
    'cv.split_j': 'int64',
    }
    df = df.astype(d_dtypes, errors='ignore')

    # %%
    for column in var_ind_real:
        try:
            df[column] = pd.to_numeric(df[column])
        except:
            pass

    # %%
    gb = df.groupby(by=var_ind_real)

    df_mean = gb.mean()
    df_std = gb.std()
    df_count = gb.count()
    df_max = gb.max()
    df_min = gb.min()
    
    # df_mean.loc[:,:,128]['r2'].reset_index().pivot(index='train.update.lr', columns='train.update.batch_size', values='r2')

    log.info(f'mean\n{df_mean}')
    log.info(f'max\n{df_max}')
    log.info(f'min\n{df_min}')
    log.info(f'std\n{df_std}')
    log.info(f'count\n{df_count}')

    with pd.ExcelWriter(path_save/'result.xlsx') as writer:
        df_mean.to_excel(writer, sheet_name='mean')
        df_std.to_excel(writer, sheet_name='std')
        df_count.to_excel(writer, sheet_name='count')
        df_max.to_excel(writer, sheet_name='max')
        df_min.to_excel(writer, sheet_name='min')

    # %%
    fig, ax = plot(df, 'r2', groupby=var_ind_real)
    fig.savefig(path_save/'r2.png')

    fig, ax = plot(df, 'l1', groupby=var_ind_real)
    fig.savefig(path_save/'l1.png')

    fig, ax = plot(df, 'mse', groupby=var_ind_real)
    fig.savefig(path_save/'mse.png')

    # %%

if __name__=='__main__':
    main()

# %%