from collections import deque
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import eval as E
import tools as T

# %%
log = logging.getLogger(__name__)

# %%
def isbetter(score_best: float, score: float, increase_better: bool):
    return (not increase_better and score<score_best) or (increase_better and score>score_best)

# %%
class Validation:
    def __init__(self, dataset, scorer):
        '''
        :param dataset: torch.utils.data.Dataset object or list of torch.utils.data.Dataset objects
        :param scorer: single or dict of scorer functions which take (y_hat, y) and returns a single score
        '''
        self.dataset = dataset
        self.scorer = scorer

    def step(self, node, name=None):
        y_list = []
        y_hat_list = []
        # If dataset is a list
        if type(self.dataset) == list:
            for dataset_ in self.dataset:
                results = E.test(node, dataset_)
                y_list.append(results['y'])
                y_hat_list.append(results['y_hat'])

        # Single dataset
        else:
            results = E.test(node, self.dataset)
            y_list.append(results['y'])
            y_hat_list.append(results['y_hat'])

        y, y_hat = np.concatenate(y_list, axis=0), np.concatenate(y_hat_list, axis=0)

        # Getting scores
        if type(self.scorer)==dict:
            score = {score_type: score(y_hat, y) for score_type, score in self.scorer.items()}
        else:
            score = self.scorer(y_hat, y)

        return score

    def reset(self):
        pass

class EarlyStopping(Validation):
    def __init__(self, dataset, scorer, primary_score=None, increase_better=False, discard_best=True, patience=None):
        '''
        :param dataset: dataloader or list of dataloaders
        '''
        super().__init__(dataset, scorer)
        self.increase_better = increase_better
        self.discard_best = discard_best

        if type(self.scorer)==dict and primary_score==None:
            self.primary_score=self.scorer.keys()[0] # default to 1st score function
        else:
            self.primary_score=primary_score

        if patience is None:
            self.patience=float('inf')
            self.cv_history = deque(maxlen=0)
        else:
            self.patience=patience
            self.cv_history = deque(maxlen=patience)

        self.reset()

    def step(self, node, name):
        score = super().step(node)
        if type(self.scorer)==dict:
            score_ = score[self.primary_score]
        else:
            score_ = score
        self.cv_history.append(score_)

        # Save best model
        if isbetter(score_best=self.best_score, score=score_, increase_better=self.increase_better):
        # (not self.increase_better and score_<self.best_score) or (self.increase_better and score_>self.best_score):
            log.info(f'Saved model: {name}')
            torch.save(node.model.state_dict(), name)
            if self.discard_best and self.best_model!=None:
                os.remove(self.best_model)

            self.best_score = score_
            self.best_model=name

            # Clear patience tracking
            self.cv_history.clear()

        # patience_end == True when best model did not appear for self.patience times
        patience_end = self.patience == len(self.cv_history)
        log.info(f'[EarlyStopping][patience: {len(self.cv_history)}/{self.cv_history.maxlen}]')

        return score, patience_end

    def reset(self):
        log.info('[EarlyStopping] Resetting...')
        self.cv_history.clear()
        self.best_model = None
        if self.increase_better:
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

# %%
class VDict(dict):
    def reset(self):
        for cv in self.values():
            cv.reset()

def track_score(score_tracker_dict, score_dict, x=None, label=None):
    '''
    :score_tracker_dict, score_dict: dict
    '''
    assert score_tracker_dict.keys() == score_dict.keys()
    keys = score_dict.keys()

    '''the keys may not be ordered. just use keys to reference'''
    # for k in keys:
    #     T = len(score_tracker_dict[k].x)
    #     score_tracker[k].step(T+1, score_dict[k])

    for score_tracker, score in zip(score_tracker_dict.values(), score_dict.values()):
        if x ==None:
            T = len(score_tracker.x)
            score_tracker.step(T+1, score, label=label)
        else:
            score_tracker.step(x, score, label=label)
