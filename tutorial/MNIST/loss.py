import logging

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# %%
log = logging.getLogger(__name__)

# %%
def weighted_crossentropy_loss(dataset, classes=None, **kwargs):
    # Loss function
    # y = np.array([data['y'] for data in dataset])
    y = np.array([y for y in dataset.targets])
    y_count = dict(pd.Series(y).value_counts().sort_index())
    if classes is None:
        class_weight = torch.tensor([1/y_count[c] for c in sorted(list(y_count.keys()))], dtype=torch.float)
        class_weight /= class_weight.sum()
    else:
        log.info(f'[weighted_crossentropyloss] classes found, expanding weight vector to: {len(classes)}')
        class_weight = torch.zeros(len(classes))
        assert set(list(y_count.keys())).issubset(set(classes))
        for c, y_count_ in y_count.items():
            class_weight[c] = 1/y_count_
        class_weight /= class_weight.sum()

    criterion = nn.CrossEntropyLoss(weight = class_weight)
    log.info('weighted_crossentropyloss - CrossEntropyLoss')
    log.info(f'y_count: {y_count}')
    log.info(f'CrossEntropy Weighted: {criterion.weight}')

    return criterion
