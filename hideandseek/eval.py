from copy import deepcopy as dcopy
import logging

import numpy as np
import sklearn.metrics as metrics
import torch

import tools as T
import tools.torch
import torch.utils.data as D

# %%
log = logging.getLogger(__name__)

# %%
'''
Testing functions

give x and produce y_hat, y_score, y_pred, etc...
'''
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
    import random
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    return {'worker_init_fn': seed_worker, 'generator': g}

test_type_list = [None, 'categorical', 'multihead_classification', 'autoencode']
def test(node, dataset, batch_size=64, test_type=None, test_f=None, result_dict=None, num_workers=0, amp=False):
    '''

    '''
    # Safety check
    if test_f is not None and result_dict is not None:
        assert callable(test_f) and issubclass(result_dict, dict), f'test_f must be callable and result_dict must be dict-like'
    elif test_f is None and result_dict is None:
        assert test_type in test_type_list, f'test_type must be one of {test_type_list}, received: {test_type}'
        test_f = get_test_f(test_type)
        result_dict = get_result_dict(test_type)
    else:
        raise Exception(f'test_f and result_dict must either both be provided or None, received [test_f: {test_f}][result_dict: {result_dict}]')

    model = node.model
    device = T.torch.get_device(model) # Test on the model's device
    model.eval()

    dataset, misc_temp = transform_misc(node, dataset)
    kwargs_dataloader = reproducible_worker_dict()
    test_loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, **kwargs_dataloader)

    with torch.no_grad():
        if amp:
            # Mixed precision for acceleration
            with torch.cuda.amp.autocast():
                for data in test_loader:
                    result_dict = test_f(data=data, model=model, result_dict=result_dict, device=device)
        else:
            for data in test_loader:
                result_dict = test_f(data=data, model=model, result_dict=result_dict, device=device)

    result_dict = {k: np.concatenate(v, axis=0) if len(v)>0 else v for k, v in result_dict.items()}

    model.train()

    dataset = inverse_transform_misc(misc_temp, dataset)

    return result_dict

def get_test_f(test_type):
    assert test_type in test_type_list, f'test_type must be one of {test_type_list}, received: {test_type}'
    if test_type is None:
        return _test_base
    elif test_type == 'categorical':
        return _test_categorical
    elif test_type == 'multihead_classification':
        return _test_multihead_categorical
    elif test_type == 'autoencode':
        return _test_autoencode
    else:
        raise Exception(f'unknown test_type: {test_type}')

def get_result_dict(test_type):
    assert test_type in test_type_list, f'test_type must be one of {test_type_list}, received: {test_type}'
    if test_type is None:
        result_list = ['y', 'y_hat']
    elif test_type in ['categorical', 'multihead_classification']:
        result_list = ['y', 'y_score', 'y_pred']
    elif test_type == 'autoencode':
        result_list = ['x', 'z', 'x_hat']
    else:
        raise Exception(f'unknown test_type: {test_type}')
    return {r:[] for r in result_list}

def _test_base(data, model, result_dict, device=None):
    # device = device if device is not None else T.torch.get_device(model)
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = model(x)

    result_dict['y'].append(y.cpu().numpy())
    result_dict['y_hat'].append(y_hat.cpu().numpy())

    return result_dict

def _test_categorical(data, model, result_dict, device=None):
    # device = device if device is not None else T.torch.get_device(model)
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_score = torch.softmax(model(x), dim=1) # (N, n_classes)
    y_score = y_score.cpu().numpy()

    result_dict['y'].append(y.cpu().numpy())
    result_dict['y_score'].append(y_score)
    result_dict['y_pred'].append(y_score.argmax(axis=-1))

    return result_dict

def _test_multihead_categorical(data, model, result_dict, device=None):
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_score = torch.softmax(model(x), dim=-1) # (N, subtype, n_classes)
    y_score = y_score.cpu().numpy()

    result_dict['y'].append(y.cpu().numpy())
    result_dict['y_score'].append(y_score)
    result_dict['y_pred'].append(y_score.argmax(axis=-1))

    return result_dict

def _test_autoencode(data, model, result_dict, device=None):
    x = data['x'].to(device)
    z = model.encoder(x)
    x_hat = torch.sigmoid(model.decoder(z))

    x, z, x_hat = x.cpu().numpy(), z.cpu().numpy(), x_hat.cpu().numpy()

    result_dict['x'].append(x)
    result_dict['z'].append(z)
    result_dict['x_hat'].append(x_hat)

    return result_dict

# %%
'''
Scorers

Individual scores
'''
# Regresison scorers
def l1_score(y_hat, y):
    return metrics.mean_absolute_error(y, y_hat)

def mse_score(y_hat, y):
    return metrics.mean_squared_error(y, y_hat)

def r2_score(y_hat, y):
    return metrics.r2_score(y, y_hat)

def p_norm(y_hat, y, p=2):
    score = ((np.abs(y_hat-y)**p).sum())**(1/p)
    return score

def accuracy_score(y_pred, y):
    score = metrics.accuracy_score(y, y_pred)
    return score

# Classification score
def sensitivity_score(y_true, y_pred):
    (tn, fp), (fn, tp) = metrics.confusion_matrix(y_true, y_pred)
    if (tp+fn)==0:
        warnings.warn('invalid value in sensitivity_score, setting to 0.0')
        return 0
    return tp / (tp+fn)
def specificity_score(y_true, y_pred):
    (tn, fp), (fn, tp) = metrics.confusion_matrix(y_true, y_pred)
    if (tn+fp)==0:
        warnings.warn('invalid value in specificity_score, setting to 0.0')
        return 0
    return tn / (tn+fp)

def classification_report_full(y_true, y_pred, discard_ovr=False):
    result = metrics.classification_report(y_true, y_pred, output_dict=True)
    result_ovr = {k:dcopy(v) for k, v in result.items() if k.isnumeric()}
    result_all = {k:dcopy(v) for k, v in result.items() if not k.isnumeric()}
    more_scorers = {'sensitivity': sensitivity_score, 'specificity': specificity_score, 'accuracy': metrics.accuracy_score} # Optimize to reduce redundant computations?
    # Additional metrics
    for c in result_ovr.keys():
        c_int = int(c)
        y_true_, y_pred_ = y_true==c_int, y_pred==c_int
        for scorer_name, scorer in more_scorers.items():
            result_ovr[c][scorer_name] = scorer(y_true_, y_pred_)

    # summary
    for scorer_name, scorer in more_scorers.items():
        result_all['macro avg'][scorer_name] = np.mean([result_ovr_[scorer_name] for result_ovr_ in result_ovr.values()])
        result_all['weighted avg'][scorer_name] = np.sum([result_ovr_[scorer_name]*result_ovr_['support'] for result_ovr_ in result_ovr.values()]) / result_all['weighted avg']['support']

    if discard_ovr:
        return result_all
    else:
        result_ovr.update(result_all)
        return result_ovr

# Multihead classification score
def multihead_accuracy_score(y_pred, y):
    '''
    :param y_pred: array of shape (N, subtype)
    :param y: array of shape (N, subtype)
    '''
    assert y_pred.ndim==2 and y.ndim==2
    score = [accuracy_score(y_pred_, y_) for y_pred_, y_ in zip(y_pred.T, y.T)]
    score = np.mean(score)
    return score

def multihead_accuracy_score_score(y_score, y):
    '''
    :param y_pred: array of shape (N, subtype, n_classes)
    :param y: array of shape (N, subtype)
    '''
    assert y_score.ndim==3 and y.ndim==2
    y_pred = y_score.argmax(axis=-1)
    return multihead_accuracy_score(y_pred, y)

# Wrapper function for hydra.utils.instantiate
class Score_wrapper:
    '''
    wraps around scorers to:
    1. instantiate an object using hydra.utils.instantiate()
    2. adjust scorer functions such that raw input can be delivered to scorer functions (y_pred or y_score can be delivered to any type of scorers)
    '''
    f_dict = {
    # regression
    'l1_score': l1_score,
    'mse_score': mse_score,
    'r2_score': r2_score,

    # multihead classification
    'multihead_accuracy_score': multihead_accuracy_score_score,
    }
    def __init__(self, func):
        self.func = self.f_dict[func]

    def __call__(self, y_hat, y):
        return self.func(y_hat, y)

# %%
# All scores for each task type
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

def classification_score(result):
    '''
    :param result: dict with the following keys: [y_pred, y_score, y]


    y_score: array of shape (N, n_classes)
    y: array of shape (N,)
    '''
    y_pred, y_score, y_true = result['y_pred'], result['y_score'], result['y']

    classification_result = classification_report_full(y_true, y_pred, discard_ovr=True)
    auroc = roc_auc_score(y_true, y_score, multi_class='ovr')
    kappa = cohen_kappa_score(y_true, y_pred)
    m_cc = matthews_corrcoef(y_true, y_pred)
    c_matrix = confusion_matrix(y_true, y_pred)

    scores = {
    'auroc': auroc,
    'kappa': kappa,
    'm_cc': m_cc,
    'c_matrix': c_matrix,
    }
    classification_result.update(scores)
    return scores

def multihead_classification_score(result):
    '''
    :param result: dict with the following keys: [y_pred, y_score, y]


    y_score: array of shape (N, subtype, n_classes)
    y: array of shape (N, subtype)
    '''
    y_pred, y_score, y = result['y_pred'], result['y_score'], result['y']
    multihead_accuracy = multihead_accuracy_score(y_pred, y)

    scores = {
    'multihead_accuracy_score': multihead_accuracy
    }
    return scores

# %%
def save_score():
    f
