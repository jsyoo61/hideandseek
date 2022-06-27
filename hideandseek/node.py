'''
might be deprecated. Think this whole validation (testing) can be integrated into eval.py
'''
from collections import deque
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import eval as E
import tools as T
import tools.modules

# %%
log = logging.getLogger(__name__)

# %%
def isbetter(score_best: float, score: float, increase_better: bool):
    return (not increase_better and score<score_best) or (increase_better and score>score_best)

# %%
'''
Validation object specifies a single dataset.
scorer is tied to each validation object.


'''

class Validation:
    def __init__(self, dataset, d_scorer, batch_size=64):
        '''
        :param dataset: torch.utils.data.Dataset object or list of torch.utils.data.Dataset objects
        :param d_scorer: dict of hideandseek scorer functions which take result(dict) and returns a single score
            # TODO: May support returning multiple scores for optimization
        '''
        self.dataset = dataset
        self.d_scorer = d_scorer
        self.batch_size = batch_size

        self.history = {score_type: T.modules.ValueTracker() for score_type in self.d_scorer.keys()}

        self.reset()

    def step(self, node):
        amp = node.amp

        ld_results = []
        # If dataset is a list
        if type(self.dataset) == list:
            for dataset_ in self.dataset:
                d_results = E.test_node(node, dataset_, batch_size=self.batch_size, amp=amp)
                ld_results.append(d_results)
            d_results_merged = T.merge_dict(ld_results)
            results = {key: np.concatenate(results, axis=0) for key, results in d_results_merged}
        # Single dataset
        else:
            results = E.test_node(node, self.dataset, batch_size=self.batch_size, amp=amp)

        # Compute scores using hideandseek eval functions which receives results (dictionary)
        score = {score_type: scorer(results) for score_type, scorer in self.d_scorer.items()}
        for score_type, valuetracker in self.history.items():
            valuetracker.step(node.iter, score[score_type])

        return score

    def plot(self, ax=None, square=False):
        '''
        plot validation history
        '''
        if ax is None:
            if square:
                nrowcol = int(np.ceil(np.sqrt(len(self.d_scorer))))
                fig, ax = plt.subplots(nrows=nrowcol, ncols=nrowcol)
            else:
                fig, ax = plt.subplots(ncols=len(self.d_scorer))
            ax = ax.flatten()
        for (score_type, valuetracker), ax_ in zip(self.history.items(), ax):
            valuetracker.plot(ax=ax_)
            ax_.set_title(score_type)
        return ax

    def reset(self):
        pass

class EarlyStopper():
    def __repr__(self):
        return f'<EarlyStopper>\npatience: {self.patience}\nincrease_better: {self.increase_better}'

    def __init__(self, increase_better, patience, primary_score=None, target_validation='default', discard_best=True):
        self.increase_better = increase_better
        self.patience = patience
        self.primary_score = primary_score
        self.target_validation = target_validation
        self.discard_best = discard_best

        self.history = T.modules.ValueTracker() # For now, only track single score
        self.reset()

    def step(self, node, score_summary, path):
        '''
        score_summary: 2-level nested dictionary of scores.
            1st-level keys are validation object names,
            2nd-level keys are score names
        '''
        score_target_validation = score_summary[self.target_validation]
        if self.primary_score is None:
            assert len(score_target_validation)==1, f"primary_score must be specified when there's more than 1 scorer functions, received: {len(score_target_validation)}"
            score = list(score_target_validation.values())[0]
            # assert type(score) is not dict, 'when primary_score is not given, the given score must be a scalar'
        else:
            score = score_target_validation[self.primary_score]
        self.history.step(node.iter, score)

        # Save best model
        if isbetter(score_best=self.best_score, score=score, increase_better=self.increase_better):
            log.info(f'[EarlyStopping]New best score: {self.best_score} -> {score}')
            log.info(f'[EarlyStopping]Saved model: {path}')
            torch.save(node.model.state_dict(), path)
            if self.discard_best and self.best_model!=None:
                os.remove(self.best_model)

            self.best_score = score
            self.best_model=path

            # Clear patience tracking
            self.history.reset()

        # patience_end == True when best model did not appear for self.patience times
        patience_end = self.patience == len(self.history)
        log.info(f'[EarlyStopping][patience: {len(self.history)}/{self.patience}]')

        return patience_end

    def reset(self):
        # log.info('[EarlyStopper] Resetting...')
        self.history.reset()
        self.best_model = None
        if self.increase_better:
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

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
        for v in self.values():
            v.reset()

    def plot(self, axes, savepath='.'):
        """
        Plot per score.
        Each axis corresponds to one scorer.
        In each axis, multiple lines correspond to multiple validation objects (datasets).
        """
        # Plot validation
        for score_type, ax in zip(d_scorer.keys(), axes.flatten()):
            P.plot_values([v.history[score_type] for v in self.values()], ax=ax)
            legend = [score_type+suffix for score_type, suffix in it.product(self.keys(), ['','_smoothed'])]
            ax.set_title(score_type)
            ax.legend(legend)
        fig.savefig(os.path.join(savepath,'validation.png'))
        return ax

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
