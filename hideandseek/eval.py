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
def transfer_misc(node, dataset, verbose=False):
    misc_temp = T.TDict()
    transfer_log = []

    if 'get_f' in node.misc:
        transfer_log.append('get_f')
        misc_temp.get_f, dataset.get_f = dataset.get_f, node.misc.get_f

    if len(transfer_log)!=0:
        log.info(f'Transferring misc from node ({node.name}) -> dataset: {transfer_log}')

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

targets_type_list = [None, 'categorical', 'multihead_classification', 'autoencode'] # move this to utils?
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
        test_f = get_test_f(targets_type)
        result_dict = get_result_dict(targets_type, keep_x=keep_x)
    else:
        raise Exception(f'test_f and result_dict must either both be provided or None, received [test_f: {test_f}][result_dict: {result_dict}]')
    return test_f, result_dict

def test(model, dataset, batch_size=64, targets_type=None, test_f=None, result_dict=None, keep_x=False, num_workers=0, amp=False):
    '''
    inference stage of nn model.
    '''
    test_f, result_dict = _test_assertion(dataset=dataset, targets_type=targets_type, test_f=test_f, result_dict=result_dict, keep_x=keep_x)

    device = T.torch.get_device(model) # Test on the model's device
    model.eval()

    # dataset, misc_temp = transfer_misc(node, dataset) # Assume the dataset is consistent
    kwargs_dataloader = reproducible_worker_dict()
    test_loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, **kwargs_dataloader)

    with torch.no_grad():
        # Mixed precision for acceleration
        if amp:
            with torch.cuda.amp.autocast():
                for data in test_loader:
                    result_dict = test_f(data=data, model=model, result_dict=result_dict, device=device, keep_x=keep_x)
        else:
            for data in test_loader:
                result_dict = test_f(data=data, model=model, result_dict=result_dict, device=device, keep_x=keep_x)

    result_dict = {k: np.concatenate(v, axis=0) if len(v)>0 else v for k, v in result_dict.items()}

    model.train() # Is this necessary?


    return result_dict

def test_node(node, dataset, batch_size=64, targets_type=None, test_f=None, result_dict=None, keep_x=False, num_workers=0, amp=False):
    '''
    wrapper around node.
    transfers get_f (preprocessing modules) of node to dataset, and returns them to original dataset after inference.
    '''
    model = node.model
    dataset, misc_temp = transfer_misc(node, dataset)
    result_dict = test(model, dataset, batch_size, targets_type, test_f, result_dict, keep_x, num_workers, amp)
    dataset = inverse_transfer_misc(misc_temp, dataset)

    return result_dict

def get_test_f(targets_type):
    assert targets_type in targets_type_list, f'targets_type must be one of {targets_type_list}, received: {targets_type}'
    if targets_type is None:
        return _test_base
    elif targets_type == 'categorical':
        return _test_categorical
    elif targets_type == 'multihead_classification':
        return _test_multihead_categorical
    elif targets_type == 'autoencode':
        return _test_autoencode
    else:
        raise Exception(f'unknown targets_type: {targets_type}')

def get_result_dict(targets_type, keep_x=False):
    assert targets_type in targets_type_list, f'targets_type must be one of {targets_type_list}, received: {targets_type}'
    if targets_type is None:
        result_list = ['y_true', 'y_hat']
        if keep_x: result_list.append('x')
    elif targets_type in ['categorical', 'multihead_classification']:
        result_list = ['y_true', 'y_hat', 'y_score', 'y_pred']
        if keep_x: result_list.append('x')
    elif targets_type == 'autoencode':
        result_list = ['x', 'z', 'x_hat']
    else:
        raise Exception(f'unknown targets_type: {targets_type}')
    return {r:[] for r in result_list}

def _test_base(data, model, result_dict, device=None, keep_x=False):
    # device = device if device is not None else T.torch.get_device(model)
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = model(x)

    result_dict['y_true'].append(y.cpu().numpy())
    result_dict['y_hat'].append(y_hat.cpu().numpy())
    if keep_x: result_dict['x'].append(x.cpu().numpy())
    return result_dict

def _test_categorical(data, model, result_dict, device=None, keep_x=False):
    # Is y_hat necessary? for torch_crossentropyloss? any other methods?
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = model(x)
    y_score = torch.softmax(y_hat, dim=1) # (N, n_classes)

    result_dict['y_true'].append(y.cpu().numpy())
    result_dict['y_hat'].append(y_hat.cpu().numpy())
    result_dict['y_score'].append(y_score.cpu().numpy())
    result_dict['y_pred'].append(y_score.argmax(axis=-1).cpu().numpy())
    if keep_x: result_dict['x'].append(x.cpu().numpy())
    return result_dict

def _test_multihead_categorical(data, model, result_dict, device=None, keep_x=False):
    x = data['x'].to(device)
    y = data['y'].to(device)
    y_hat = model(x)
    y_score = torch.softmax(y_hat, dim=-1) # (N, subtype, n_classes)

    result_dict['y_true'].append(y.cpu().numpy())
    result_dict['y_hat'].append(y_hat.cpu().numpy())
    result_dict['y_score'].append(y_score.cpu().numpy())
    result_dict['y_pred'].append(y_score.argmax(axis=-1).cpu().numpy())

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

They all receive dict of numpy arrays.
Predefined keys are:
'y_true', 'y_hat', 'y_pred'

Example
-------
result = {'y_true': y_true, 'y_pred': y_pred}
classification_score(result)


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
    # (tn, fp), (fn, tp) = metrics.confusion_matrix(result['y_true'], y_pred)
    # if (tp+fn)==0:
    #     warnings.warn('invalid value in sensitivity_score, setting to 0.0')
    #     return 0
    # return tp / (tp+fn)

def specificity_score(result):
    (tn, fp), (fn, tp) = metrics.confusion_matrix(result['y_true'], y_pred)
    if (tn+fp)==0:
        warnings.warn('invalid value in specificity_score, setting to 0.0')
        return 0
    return tn / (tn+fp)

# Multiclass classification
def classification_report_full(result, discard_ovr=False):
    '''
    Adds additional metrics to sklearn.classification_report
    '''
    y_true, y_pred = result['y_true'], result['y_pred']
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

#
# # Regresison scorers
# def l1_score(y_true, y_hat): # alias
#     return metrics.mean_absolute_error(y_true, y_hat)
#
# def l2_score(y_true, y_hat): # alias
#     return metrics.mean_squared_error(y_true, y_hat)
#
# def mse_score(y_true, y_hat): # alias
#     return metrics.mean_squared_error(y_true, y_hat)
#
# def r2_score(y_true, y_hat):
#     return metrics.r2_score(y_true, y_hat)
#
# def p_norm_score(y_true, y_hat, p=2): # When using this with Validation object, use functools.partial() to fix ``p``
#     score = ((np.abs(y_hat-y_true)**p).sum())**(1/p)
#     return score
#
# # Classification score
# def accuracy_score(y_true, y_pred):
#     score = metrics.accuracy_score(y_true, y_pred)
#     return score
#
# def accuracy_score_yhat(y_true, y_hat): # Use this for Validation object, which feeds y_hat to all of its scorers.
#     y_pred = y_hat.argmax(axis=1)
#     score = metrics.accuracy_score(y_true, y_pred)
#     return score
#
# # Binary classification
# def sensitivity_score(y_true, y_pred, **kwargs): # alias
#     '''sensitivity == recall == tpr'''
#     return metrics.recall_score(y_true, y_pred, **kwargs)
#     # (tn, fp), (fn, tp) = metrics.confusion_matrix(y_true, y_pred)
#     # if (tp+fn)==0:
#     #     warnings.warn('invalid value in sensitivity_score, setting to 0.0')
#     #     return 0
#     # return tp / (tp+fn)
#
# def specificity_score(y_true, y_pred):
#     (tn, fp), (fn, tp) = metrics.confusion_matrix(y_true, y_pred)
#     if (tn+fp)==0:
#         warnings.warn('invalid value in specificity_score, setting to 0.0')
#         return 0
#     return tn / (tn+fp)
#
# def _pr_score(y_true, y_score, threshold):
#     y_pred = T.numpy.binarize(y_score, threshold)
#     pr = metrics.precision_score(y_true, y_pred)
#     rec = metrics.recall_score(y_true, y_pred)
#     return pr, rec
#
# def precision_recall_curve_all(y_true, y_score, num_workers=None):
#     '''
#     compute precision-recall curve for all threshold.
#     Because sklearn.metrics.precision_recall_curve does not return full set of thresholds
#     sklearn drops the results after when recall hits 1.
#
#     -> is this function really necessary?
#     '''
#     thresholds = np.sort(np.unique(y_score)) # increasing threshold
#     thresholds = np.append(thresholds,1) # Since tnp.binarize binarizes with y_pred[y_score>=threshold] == 1
#     if num_workers==0:
#         prs, recs = [], []
#         for t in thresholds:
#             y_pred = T.numpy.binarize(y_score, threshold)
#             prs.append(metrics.precision_score(y_true, y_pred))
#             recs.append(metrics.recall_score(y_true, y_pred))
#     elif num_workers is None or T.isint(num_workers):
#         with multiprocessing.Pool() as p:
#             l_pr = p.starmap(pr_score, zip(it.repeat(y_true, len(thresholds)), it.repeat(y_score, len(thresholds)), thresholds))
#
#
#     return prs, recs, thresholds
#
# # Multiclass classification
# def classification_report_full(y_true, y_pred, discard_ovr=False):
#     '''
#     Adds additional metrics to sklearn.classification_report
#     '''
#     result = metrics.classification_report(y_true, y_pred, output_dict=True)
#     result_ovr = {k:dcopy(v) for k, v in result.items() if k.isnumeric()}
#     result_all = {k:dcopy(v) for k, v in result.items() if not k.isnumeric()}
#     more_scorers = {'sensitivity': sensitivity_score, 'specificity': specificity_score, 'accuracy': metrics.accuracy_score} # Optimize to reduce redundant computations?
#     # Additional metrics
#     for c in result_ovr.keys():
#         c_int = int(c)
#         y_true_, y_pred_ = y_true==c_int, y_pred==c_int
#         for scorer_name, scorer in more_scorers.items():
#             result_ovr[c][scorer_name] = scorer(y_true_, y_pred_)
#
#     # summary
#     for scorer_name, scorer in more_scorers.items():
#         result_all['macro avg'][scorer_name] = np.mean([result_ovr_[scorer_name] for result_ovr_ in result_ovr.values()])
#         result_all['weighted avg'][scorer_name] = np.sum([result_ovr_[scorer_name]*result_ovr_['support'] for result_ovr_ in result_ovr.values()]) / result_all['weighted avg']['support']
#
#     if discard_ovr:
#         return result_all
#     else:
#         result_ovr.update(result_all)
#         return result_ovr
#
# # Multihead classification score
# def multihead_accuracy_score(y_pred, y):
#     '''
#     :param y_pred: array of shape (N, subtype)
#     :param y: array of shape (N, subtype)
#     '''
#     assert y_pred.ndim==2 and y.ndim==2
#     score = [accuracy_score(y_pred_, y_) for y_pred_, y_ in zip(y_pred.T, y.T)]
#     score = np.mean(score)
#     return score
#
# def multihead_accuracy_score_score(y_score, y):
#     '''
#     :param y_pred: array of shape (N, subtype, n_classes)
#     :param y: array of shape (N, subtype)
#     '''
#     assert y_score.ndim==3 and y.ndim==2
#     y_pred = y_score.argmax(axis=-1)
#     return multihead_accuracy_score(y_pred, y)
