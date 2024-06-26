import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

import tools as T
import tools.torch

# %%
log = logging.getLogger(__name__)

# %%
'''
feature visualization module
'''


"""

Parameters
----------
objective : nn.Module object which takes an input and outputs a single value to optimize.
    Recommended way to make this object is to build another class inheriting nn.Module,
    and constructing the objective function within the class.

See Also
--------
Output_i : nn.Module which computes i-th output of the given model.
    Useful to construct simple objective function.

Examples
--------
>>>
"""

class Output_i(nn.Module):
    """
    Output objective

    Parameters
    ----------
    model : nn.Module object
        Model to optimize.
    class_i: int
        Output class index.
    maximize: bool, default=True
        If True, then forward returns negated value (since loss is minimized in pytorch),
    suppress: bool, default=False
        If True, then forward adds non-negated values of other classes.
        Suppresses other class activations.

    Examples
    --------
    >>> log.info('hello')
    hello
    """
    def __init__(self, model, class_i, maximize=True, suppress=False):
        super().__init__()
        self.model = model
        self.class_i = class_i
        self.maximize = maximize
        self.suppress = suppress

    def forward(self, x):
        y_hat = self.model(x)
        objective = y_hat[:, self.class_i]
        if self.suppress:
            class_i_suppress = list(range(y_hat.shape[-1]))
            class_i_suppress.remove(self.class_i)
            objective = objective - y_hat[:, class_i_suppress].mean(-1)

        if self.maximize:
            objective = -objective
        # if self.maximize:
        #     objective = -y_hat[:, self.class_i]
        # else:
        #     objective = y_hat[:, self.class_i]
        return objective

class RandomSampler():
    '''
    Sample random data
    '''
    allowed_methods = ['rand', 'randn']
    def __init__(self, input_shape, method='rand'):
        self.input_shape = input_shape
        self.method = method
        assert method in self.allowed_methods, f'allowed methods are: {self.allowed_methods}\nreceived: {method}'

    def sample(self, n_sample, method=None):
        method = self.method if method is None else method
        if method == 'rand':
            sample = torch.rand(n_sample, *self.input_shape)
        elif method == 'randn':
            sample = torch.randn(n_sample, *self.input_shape)
        return sample

# %%
def _optimize(objective, sample, updates, Optimizer, optimizer_kwargs, preprocess=None, regularization=None, threshold=None, return_best=False, verbose=False):
    device_original = sample.device
    device = T.torch.get_device(objective)
    sample = sample.clone().detach().to(device) # clone sample so that original tensor will not be modified.
    sample.requires_grad_(True)

    # if not sample.requires_grad:
    #     warnings.warn('Provide sample with sample.requires_grad_(True).\
    #     _optmize() function will automatically setting requires_grad to True, but this may resuilt in undesired results.')

    optimizer_ = Optimizer([sample], **optimizer_kwargs)

    if preprocess is not None:
        log.info('preprocess specified')
    if regularization is not None:
        log.info('regularization specified')

    if verbose:
        with torch.no_grad():
            sample_processed = sample if preprocess is None else preprocess(sample)
            sample_regularized = sample if regularization is None else regularization(sample_processed)
            y_hat = objective(sample_regularized)
            loss = y_hat.mean()
            log.info(f'[iter: 0/{updates}][loss: {loss.item()}]')

    n_verbose = updates // 10

    if threshold is not None:
        threshold_crossed_i = np.zeros(len(sample), dtype=np.bool)
        sample_optimized = torch.empty_like(sample)

    for i in range(1, updates+1):
        sample_processed = sample if preprocess is None else preprocess(sample)
        sample_regularized = sample if regularization is None else regularization(sample_processed)
        y_hat = objective(sample_regularized)
        loss = y_hat.mean()

        # Note this is threshold after preprocessing & regularization(!!)
        if threshold is not None:
            with torch.no_grad():
                # Store sample when threshold passed
                threshold_crossed_i_ = (y_hat.detach().cpu().numpy() < threshold)
                store_i = ~threshold_crossed_i&threshold_crossed_i_
                sample_optimized[store_i] = sample[store_i].detach()
                threshold_crossed_i[store_i] = True

                if store_i.any():
                    log.info('threshold!')
                    log.info(store_i, sample_optimized[store_i].detach(), sample_regularized[store_i].detach(), y_hat)
                # # if threshold_crossed_i_.any():
                # #     log.info('threshold!')
                # #     break

                if threshold_crossed_i_.all():
                    log.info(f'Threshold crossed. [-loss: ({-loss.item()}) > threshold: ({threshold})]')
                    break

        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()
        if i%n_verbose==0:
            log.info(f'[iter: {i}/{updates}][loss: {loss.item()}]')

    # Should return preprocessed sample?
    if threshold is not None:
        sample_optimized[~threshold_crossed_i] = sample[~threshold_crossed_i].detach()
        return sample_optimized.to(device_original)
    else:
        return sample.detach().to(device_original)

def optimize(objective, sample, updates, Optimizer, optimizer_kwargs, preprocess=None, regularization=None, threshold=None, return_best=False, verbose=False, batch_size=None):
    """
    Optimize sample
    Copies the sample and returns the optimized tensor.

    Parameters
    ----------
    arg :
        dd
    Returns
    -------
    sample : torch.Tensor which is optimized

    References
    ----------
    .. [1] `Feature Visualization
           <https://distill.pub/2017/feature-visualization/>`_
    Examples
    --------
    >>>
    """
    if batch_size is None:
        sample_optimized = _optimize(objective=objective, sample=sample, updates=updates, preprocess=preprocess, regularization=regularization, Optimizer=Optimizer, optimizer_kwargs=optimizer_kwargs, threshold=threshold, return_best=return_best, verbose=verbose)
    else:
        if verbose: log.info('Run in batches')
        loader = D.DataLoader(sample, batch_size=batch_size, drop_last=False)
        l_sample_optimized = []
        for sample_batch in loader:
            sample_optimized = _optimize(objective=objective, sample=sample, updates=updates, preprocess=preprocess, regularization=regularization, Optimizer=Optimizer, optimizer_kwargs=optimizer_kwargs, threshold=threshold, return_best=return_best, verbose=verbose)
            l_sample_optimized.append(sample_optimized)
        sample_optimized = torch.stack(l_sample_optimized, axis=0)
    return sample_optimized
