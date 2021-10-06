import numpy as np

import tools as T
import tools.torch

# %%
def transform_misc(node, dataset, verbose=False):
    misc_temp = T.TDict()
    transfer_log = []

    if 'get_f' in node.misc:
        transfer_log.append('get_f')
        misc_temp.get_f, dataset.get_f = dataset.get_f, node.misc.get_f

    if len(transfer_log)!=0:
        log.info(f'Transferring misc from node ({node.name}) -> dataset: {transfer_log}')

    return dataset, misc_temp

def inverse_transform_misc(misc_temp, dataset):
    transfer_log = []
    if 'get_f' in misc_temp:
        transfer_log.append('get_f')
        dataset.get_f = misc_temp.get_f

    if len(transfer_log)!=0:
        log.info(f'Inverse transferring misc -> dataset: {transfer_log}')

    return dataset

def reproducible_worker_dict():
    '''Generate separate random number generators for workers,
    so that the global random state is not consumed,
    thereby ensuring reproducibility'''
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {'worker_init_fn': seed_worker, 'generator': g}

def test(node, dataset, batch_size=64):
    '''
    :param node: hs.Node object

    '''
    model = node.model
    device = T.torch.get_device(model) # Test on the model's device
    model.eval()

    dataset, misc_temp = transform_misc(node, dataset)
    kwargs_dataloader = reproducible_worker_dict()
    test_loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs_dataloader)

    y_list = []
    y_hat_list = []
    with torch.no_grad():
        for data in test_loader:
            x = data['x'].to(device)
            y = data['y'].to(device)
            y_hat = model(x)

            y_list.append(y.cpu().numpy())
            y_hat_list.append(y_hat.cpu().numpy())
    y = np.concatenate(y_list, axis=0)
    y_hat = np.concatenate(y_hat_list, axis=0)
    model.train()

    dataset = inverse_transform_misc(misc_temp, dataset)

    results = {'y': y, 'y_hat': y_hat}
    return results

def l1_score(y_hat, y):
    return metrics.mean_absolute_error(y, y_hat)

def mse_score(y_hat, y):
    return metrics.mean_squared_error(y, y_hat)

def r2_score(y_hat, y):
    return metrics.r2_score(y, y_hat)

def p_norm(y_hat, y, p=2):
    score = ((np.abs(y_hat-y)**p).sum())**(1/p)
    return score

class Score_wrapper:
    f_dict = {
    'l1_score': l1_score,
    'mse_score': mse_score,
    'r2_score': r2_score,
    }
    def __init__(self, func):
        self.func = self.f_dict[func]

    def __call__(self, y_hat, y):
        return self.func(y_hat, y)
# %%
def regression_score(result):
    '''
    :param result: dict with the following keys: [y_hat, y]
    '''
    y_hat, y = result['y_hat'], result['y']
    l1 = l1_score(y_hat, y)
    mse = mse_score(y_hat, y)
    r2 = r2_score(y_hat, y)

    scores = {
    'l1': l1,
    'mse': mse,
    'r2': r2
    }

    return scores
