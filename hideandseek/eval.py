from copy import deepcopy as dcopy
import logging
import multiprocessing

import numpy as np
import sklearn.metrics as metrics
import torch

import tools as T
import tools.torch
import tools.numpy
import torch.utils.data as D

# %%
log = logging.getLogger(__name__)

# %%
'''
Testing functions

give x and produce y_hat, y_score, y_pred, etc...
'''
def transfer_misc(model, dataset, verbose=False):
    misc_temp = T.TDict()
    transfer_log = []

    if 'get_f' in model.misc:
        transfer_log.append('get_f')
        misc_temp.get_f, dataset.get_f = dataset.get_f, model.misc.get_f

    if len(transfer_log)!=0:
        log.info(f'Transferring misc from model ({model.name}) -> dataset: {transfer_log}')

    return dataset, misc_temp

def inverse_transfer_misc(misc_temp, dataset):
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

def _test_assertion(dataset, targets_type, test_f, result_dict, keep_x):
    # Safety check
    if test_f is not None and result_dict is not None:
        assert callable(test_f) and issubclass(type(result_dict), dict), f'test_f must be callable and result_dict must be dict-like'
    elif test_f is None and result_dict is None:
        # Infer targets_type and create corresponding test_f and result_dict
        if targets_type is None:
            if hasattr(dataset, 'targets_type'):
                targets_type = dataset.targets_type
            else:
                raise Exception('When test_f and result_dict is not given, targets_type must be given or the dataset must have the attribute "targets_type"')
        assert targets_type in targets_type_list, f'targets_type must be one of {targets_type_list}, received: {targets_type}'
        test_f, result_dict = get_test_f(targets_type, keep_x=keep_x)
    else:
        raise Exception(f'test_f and result_dict must either both be provided or None, received [test_f: {test_f}][result_dict: {result_dict}]')
    return test_f, result_dict

def test_torch(network, dataset, batch_size=64, targets_type=None, test_f=None, result_dict=None, keep_x=False, num_workers=0, amp=False):
    '''
    inference stage of network.
    '''
    test_f, result_dict = _test_assertion(dataset=dataset, targets_type=targets_type, test_f=test_f, result_dict=result_dict, keep_x=keep_x)

    device = T.torch.get_device(network) # Test on the network's device
    network.eval()

    # dataset, misc_temp = transfer_misc(node, dataset) # Assume the dataset is consistent
    kwargs_dataloader = reproducible_worker_dict()
    test_loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, **kwargs_dataloader)

    with torch.no_grad():
        # Mixed precision for acceleration
        if amp:
            with torch.autocast(device_type=device.type):
                for data in test_loader:
                    result_dict = test_f(data=data, network=network, result_dict=result_dict, device=device, keep_x=keep_x)
        else:
            for data in test_loader:
                result_dict = test_f(data=data, network=network, result_dict=result_dict, device=device, keep_x=keep_x)

    result_dict = {k: torch.cat(v, dim=0) if len(v)>0 else v for k, v in result_dict.items()}
    if amp: result_dict = {k: v.to(torch.float16) for k, v in result_dict.items()} # Could maintain float16, but numpy sometimes fails with float16.
    # if amp: result_dict = {k: v.to(torch.float32) for k, v in result_dict.items()} # Could maintain float16, but numpy sometimes fails with float16.
    result_dict = {k: v.numpy() for k, v in result_dict.items()}

    network.train() # Is this necessary?

    return result_dict

def test_model(model, dataset, batch_size=64, targets_type=None, test_f=None, result_dict=None, keep_x=False, num_workers=0, amp=False):
    '''
    wrapper around model.
    transfers get_f (preprocessing modules) of node to dataset, and returns them to original dataset after inference.
    '''
    network = model.network
    dataset, misc_temp = transfer_misc(model, dataset)
    result_dict = test_torch(network, dataset, batch_size, targets_type, test_f, result_dict, keep_x, num_workers, amp)
    dataset = inverse_transfer_misc(misc_temp, dataset)

    return result_dict

def evaluate(results, metrics):
    if type(metrics) is dict:
        scores = {}
        for metric_name, metric in metrics.items():
            scores[metric_name] = metric(results)
    elif type(metrics) is list:
        scores = [metric(results) for metric in metrics]
    else:
        scores = metrics(results)
    return scores

targets_type_list = [None, 'categorical', 'multihead_classification', 'autoencode', 'regression'] # move this to utils?
def get_test_f(targets_type, keep_x=False):
    assert targets_type in targets_type_list, f'targets_type must be one of {targets_type_list}, received: {targets_type}'
    if targets_type is None or targets_type == 'regression':
        test_f = _test_base
        result_list = ['y_true', 'y_hat']
    elif targets_type == 'categorical':
        test_f = _test_categorical
        result_list = ['y_true', 'y_hat', 'y_score', 'y_pred']
    elif targets_type == 'multihead_classification':
        test_f = _test_multihead_categorical
        result_list = ['y_true', 'y_hat', 'y_score', 'y_pred']
    elif targets_type == 'autoencode':
        test_f = _test_autoencode
        result_list = ['x', 'z', 'x_hat']
    else:
        raise Exception(f'unknown targets_type: {targets_type}')
    
    if keep_x and 'x' not in result_list: result_list.append('x')
    result_dict = {r:[] for r in result_list}

    return test_f, result_dict

def _test_base(data, network, result_dict, device=None, keep_x=False):
    # device = device if device is not None else T.torch.get_device(network)
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = network(x)

    result_dict['y_true'].append(y.cpu())
    result_dict['y_hat'].append(y_hat.cpu())
    if keep_x: result_dict['x'].append(x.cpu())
    return result_dict

def _test_categorical(data, network, result_dict, device=None, keep_x=False):
    # Is y_hat necessary? for torch_crossentropyloss? any other methods?
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = network(x)
    y_score = torch.softmax(y_hat, dim=1) # (N, n_classes)

    result_dict['y_true'].append(y.cpu())
    result_dict['y_hat'].append(y_hat.cpu())
    result_dict['y_score'].append(y_score.cpu())
    result_dict['y_pred'].append(y_score.argmax(axis=-1).cpu())
    if keep_x: result_dict['x'].append(x.cpu())
    return result_dict

def _test_multihead_categorical(data, network, result_dict, device=None, keep_x=False):
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = network(x)
    y_score = torch.softmax(y_hat, dim=-1) # (N, subtype, n_classes)

    result_dict['y_true'].append(y.cpu())
    result_dict['y_hat'].append(y_hat.cpu())
    result_dict['y_score'].append(y_score.cpu())
    result_dict['y_pred'].append(y_score.argmax(axis=-1).cpu())
    if keep_x: result_dict['x'].append(x.cpu())
    return result_dict

def _test_autoencode(data, network, result_dict, device=None):
    x = data['x'].to(device)
    z = network.encoder(x)
    x_hat = torch.sigmoid(network.decoder(z))

    x, z, x_hat = x.cpu(), z.cpu(), x_hat.cpu()

    result_dict['x'].append(x)
    result_dict['z'].append(z)
    result_dict['x_hat'].append(x_hat)
    return result_dict

# %%
'''
Scorers

They all receive dict of numpy arrays.
Predefined keys are:
'y_true', 'y_hat', 'y_pred'

Example
-------
result = {'y_true': y_true, 'y_pred': y_pred}
classification_score(result)

scores = {'scorer type': score, ...}


# order of arguments follow sklearn convention: y_true, y_hat/pred

Note
----
pytorch metrics receive (y_hat/pred, y_true), whereas sklearn receive (y_true, y_hat/pred)
'''

# Regresison scorers
def l1_score(result): # alias
    return metrics.mean_absolute_error(result['y_true'], result['y_hat'])

def l2_score(result): # alias
    return metrics.mean_squared_error(result['y_true'], result['y_hat'])

def mse_score(result): # alias
    return metrics.mean_squared_error(result['y_true'], result['y_hat'])

def r2_score(result):
    return metrics.r2_score(result['y_true'], result['y_hat'])

def p_norm_score(result, p=2): # When using this with Validation object, use functools.partial() to fix ``p``
    score = ((np.abs(result['y_hat']-result['y_true'])**p).sum())**(1/p)
    return score

# Classification score
def accuracy_score(result):
    return metrics.accuracy_score(result['y_true'], result['y_pred'])

# def accuracy_score_y_hat(result): # Use this for Validation object, which feeds y_hat to all of its scorers.
#     return metrics.accuracy_score(result['y_true'], result['y_hat'].argmax(axis=1))

# Binary classification
def sensitivity_score(result): # alias
    '''sensitivity == recall == tpr'''
    return metrics.recall_score(result['y_true'], result['y_pred'])

def specificity_score(result):
    (tn, fp), (fn, tp) = metrics.confusion_matrix(result['y_true'], result['y_pred'])
    if (tn+fp)==0:
        warnings.warn('invalid value in specificity_score, setting to 0.0')
        return 0
    return tn / (tn+fp)

# Multiclass classification
def classification_report_full(result, discard_ovr=False):
    '''
    Adds additional metrics to sklearn.classification_report
    '''
    # TODO: add y_score
    y_score = result['y_score'] if 'y_score' in result else None

    y_true, y_pred = result['y_true'], result['y_pred']
    scores = metrics.classification_report(y_true, y_pred, output_dict=True)
    scores_ovr = {k:dcopy(v) for k, v in scores.items() if k.isnumeric()}
    scores_all = {k:dcopy(v) for k, v in scores.items() if not k.isnumeric()}
    more_scorers = {'sensitivity': sensitivity_score, 'specificity': specificity_score, 'accuracy': accuracy_score} # Optimize to reduce redundant computations?

    # Additional metrics
    for c in scores_ovr.keys():
        c_int = int(c)
        y_true__c, y_pred_c = y_true==c_int, y_pred==c_int
        for scorer_name, scorer in more_scorers.items():
            scores_ovr[c][scorer_name] = scorer({'y_true': y_true__c, 'y_pred': y_pred_c})

    if y_score is not None:
        for c in scores_ovr.keys():
            c_int = int(c)
            y_score_ = y_score[:, c_int]
            metrics.roc_auc_score(y_true, y_score_)

    # summary
    for scorer_name, scorer in more_scorers.items():
        scores_all['macro avg'][scorer_name] = np.mean([scores_ovr_[scorer_name] for scores_ovr_ in scores_ovr.values()])
        scores_all['weighted avg'][scorer_name] = np.sum([scores_ovr_[scorer_name]*scores_ovr_['support'] for scores_ovr_ in scores_ovr.values()]) / scores_all['weighted avg']['support']

    if discard_ovr:
        return scores_all
    else:
        scores_ovr.update(scores_all)
        return scores_ovr

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

def classification_score(result, multi_class='ovr', discard_ovr=False):
    '''
    :param result: dict with the following keys: [y_pred, y_score, y]


    y_score: array of shape (N, n_classes)
    y: array of shape (N,)
    '''
    y_pred, y_score, y_true = result['y_pred'], result['y_score'], result['y']

    # classification_result = classification_report_full(y_true, y_pred, discard_ovr=discard_ovr)
    classification_result = classification_report_full(result, discard_ovr=discard_ovr)
    auroc = metrics.roc_auc_score(y_true, y_score, multi_class=multi_class)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    m_cc = metrics.matthews_corrcoef(y_true, y_pred)
    c_matrix = metrics.confusion_matrix(y_true, y_pred)

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
'''
May be deprecated
'''
def save_score():
    f

def _pr_score(y_true, y_score, threshold):
    y_pred = T.numpy.binarize(y_score, threshold)
    pr = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return pr, rec

def precision_recall_curve_all(y_true, y_score, num_workers=None):
    '''
    compute precision-recall curve for all threshold.
    Because sklearn.metrics.precision_recall_curve does not return full set of thresholds
    sklearn drops the results after when recall hits 1.

    -> is this function really necessary?
    -> may be deprecated
    '''
    thresholds = np.sort(np.unique(y_score)) # increasing threshold
    thresholds = np.append(thresholds,1) # Since tnp.binarize binarizes with y_pred[y_score>=threshold] == 1
    if num_workers==0:
        prs, recs = [], []
        for t in thresholds:
            y_pred = T.numpy.binarize(y_score, threshold)
            prs.append(metrics.precision_score(y_true, y_pred))
            recs.append(metrics.recall_score(y_true, y_pred))
    elif num_workers is None or T.isint(num_workers):
        with multiprocessing.Pool() as p:
            l_pr = p.starmap(_pr_score, zip(it.repeat(y_true, len(thresholds)), it.repeat(y_score, len(thresholds)), thresholds))

    return prs, recs, thresholds
