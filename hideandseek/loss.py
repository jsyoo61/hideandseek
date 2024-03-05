import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

def weighted_crossentropy_loss(dataset, classes=None):
    """
    torch.nn.CrosEntropyLoss with class weights inferred from given dataset.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset object.
        If the dataset supports .get_y() function or .get_y_all() function,
        all y values are retreived using the methods. If the dataset does not support such functios,
        [data['y'] for data in dataset] is called which may be very slow if loading data['x'] is heavy.
    classes: list of all possible classes, default=None
        If the dataset argument does not have any instances of some class,
        classes argument can be given to ensure propose size of class weights are applied.
        Missing labels are weighted with 0.
    Returns
    -------
    criterion: nn.CrossEntropyLoss object
        nn.CrossEntropyLoss with class weights applied
    """
    # Loss function
    if hasattr(dataset, 'get_y_all'):
        y = dataset.get_y_all()
    elif hasattr(dataset, 'get_y'):
        y = torch.stack([dataset.get_y(idx) for idx in range(len(dataset))])
    else:
        data_sample = next(iter(dataset))
        if isinstance(data_sample, dict):
            y = torch.stack([data['y'] for data in dataset])
        elif isinstance(data_sample, tuple) and len(data_sample)==2:
            y = torch.stack([data[1] for data in dataset])
        else:
            raise Exception(f'unknown dataset return type: {type(data_sample)}')
    y_unique, y_count = torch.unique(y, return_counts=True, sorted=True) # sorted by default

    if classes is None:
        class_weight = 1/y_count
    else:
        log.info(f'[weighted_crossentropyloss] classes found, expanding weight vector to: {len(classes)}')
        class_weight = torch.zeros(len(classes))
        assert set(y_unique).issubset(set(classes))
        for c, y_count_ in zip(y_unique, y_count):
            class_weight[c] = 1/y_count_ 
    class_weight /= class_weight.sum()

    criterion = nn.CrossEntropyLoss(weight = class_weight)
    log.info('Weighted_crossentropyloss - CrossEntropyLoss')
    log.info(f'y_count: {y_count}')
    log.info(f'CrossEntropy Weighted: {criterion.weight}')

    return criterion