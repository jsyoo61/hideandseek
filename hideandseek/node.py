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
        self.iter = 0
        self.n_batch=0

    def print(self, content):
        if self.verbose:
            print(content)
        else:
            log.info(content)

    def set_misc(self):
        '''Set miscellaneous variables'''
        self.misc = tools.TDict()
        if self.dataset is not None:
            if hasattr(self.dataset, 'get_f'): # get_f equal to preprocessing pipeline before deep learning model
                self.print('get_f found in loader.dataset')
                self.misc.get_f = self.dataset.get_f

    def validate(self):
        if self.validation is not None:
            score_summary = {}
            for cv_name, validation in self.validation.items(): # Different types of datasets
                score = validation.step(self)
                score_summary[cv_name] = score
                self.print(f'[cv_name: {cv_name}] Validation Score: {score}')
            if self.earlystopper is not None:
                patience_end = self.earlystopper.step(self, score_summary, os.path.join(self.model_dir, f'model_{self.iter}.pt'))
                if patience_end:
                    self.print('patience met, flushing earlystopper history')
                    self.earlystopper.history.reset()
            else:
                patience_end = False
        return patience_end

    # def validate(self, prefix=''):
    #     # TODO: Is prefix really necessary?
    #     patience_end = False
    #     if self.validation is not None:
    #         for cv_name, validation in self.validation.items():
    #             score, _patience_end = self._validation(validation)
    #             # This exists for the case when there are multiple EarlyStopping objects. If patience_end occurs a single time, return True
    #             if _patience_end:
    #                 patience_end=True
    #             self.print(f'{prefix}[cv_name: {cv_name}] Validation Score: {score}')
    #             V.track_score(self.cv_tracker[cv_name], score, x=self.iter, label=self.name)
    #     return patience_end
    #
    # def _validation(self, validation):
    #     patience_end = False
    #     if type(validation)==V.EarlyStopping:
    #         # score, patience_end = validation.step(self, os.path.join(self.model_dir, f'model_{self.iter}.pt'))
    #         score, patience_end = validation.step(self)
    #         if patience_end:
    #             self.print('patience met, flushing cv_history')
    #             validation.cv_history.clear()
    #     else:
    #         # score = validation.step(self, os.path.join(self.model_dir, f'model_{self.iter}.pt'))
    #         score = validation.step(self)
    #     return score, patience_end

    def generate_loader(self):
        if self.dataset is not None:
            # To avoid batch normalization layers from raising exceptions
            self.loader = D.DataLoader(self.dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=True, **U.reproducible_worker_dict()) if len(self.dataset) % self.cfg_train['batch_size'] == 1 \
                    else D.DataLoader(self.dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=False, **U.reproducible_worker_dict())
        else:
            log.warning('No dataset found. Skipping generate_loader()')

    def train(self, epoch=None, new_op=True, no_val=False, step=None):
        '''
        Trains the model with the specified duration.
        '''
        assert epoch is None or step is None, f'only one of epoch or step  can be specified. received epoch: {epoch}, step: {step}'
        if step is None:
            horizon = 'epoch'
            if epoch is None: # When neither epoch or step are specified
                assert 'epoch' in self.cfg_train, 'key "epoch" must be provided in cfg_train when argument "epoch" is not specified.'
                T = self.cfg_train['epoch']
            else:
                T = epoch
        else:
            assert epoch is None, f'Either epoch or step must be specified. Received epoch: {epoch}, step: {step}'
            horizon = 'step'
            T = updates
        self.print(f'[Node: {self.name}][horizon: {horizon}][duration: {T}]')

        self.model.train()
        device = tools.torch.get_device(self.model)
        self.criterion = self.criterion.to(device)
        self.loss_tracker.reset()
        self.generate_loader()
        '''
        There may be one or more loaders, but self.loader is the standard of synchronization
        Either return multiple values from dataset, or modify self.forward to use other loaders
        '''

        if self.validation is not None:
            if 'cv_step' not in self.cfg_train:
                self.print('Node.validation given, but "cv_step" not specified in cfg_train. Defaults to 1 epoch')
                self.cfg_train['cv_step'] = len(self.loader)
            self.validation.reset()
            self.validate() # initial testing

        # Make new optimizer
        if new_op or not hasattr(self, 'op'):
            if not hasattr(self, 'op') and not new_op:
                log.warning("new_op=False when there's no pre-existing optimizer. Ignoring new_op...")
            # Weight decay optional
            self.op = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'], weight_decay=self.cfg_train['weight_decay']) if 'weight_decay' in self.cfg_train \
                        else optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
            # self.op = optim.Adam(self.model.parameters(), **self.cfg_train)

        if horizon == 'epoch':
            self._step_epoch(T=T, no_val=no_val, device=device)
        elif horizon=='step':
            self._step_step(T=T, no_val=no_val, device=device)
        # else:
        #     raise Exception(f'Invalid horizon type: {horizon}')

        self.criterion = self.criterion.cpu()
        return self.loss_tracker

    def _step_epoch(self, T, no_val=False, device=None):
        device = device if device is not None else tools.torch.get_device(self.model)

        _iter = 0
        for epoch in range(1, T+1):
            self.train_meter.reset()
            self.epoch_f()
            for batch_i, data in enumerate(self.loader, 1):
                self.iter += 1
                _iter += 1
                loss = self.update(data, device)
                self.print(f'[iter_sum: {self.iter}][Epoch: {epoch}/{T}][Batch: {batch_i}/{len(self.loader)}][Loss: {loss.item():.7f} (Avg: {self.train_meter.avg:.7f})]')
                self.loss_tracker.step(self.iter, loss)

                # Validation
                if (self.validation is not None) and (not no_val) and (_iter % self.cfg_train['cv_step']==0):
                    patience_end = self.validate()
                    if patience_end: # If patience has reached, stop training
                        self.print('Patience met, stopping training')
                        return None # return None since double break is impossible in python

    def _step_step(self, T, no_val=False, device=None):
        device = device if device is not None else tools.torch.get_device(self.model)
        if hasattr(self, '_loader_inst'): del self._loader_inst

        for i in range(1, T+1):
            # Get Data
            try:
                if hasattr(self, '_loader_inst'):
                    data = next(self._loader_inst)
                    self.n_batch += 1
                else:
                    raise StopIteration
            except StopIteration as e:
                self.train_meter.reset()
                self.epoch_f()
                self._loader_inst = iter(self.loader)
                data = next(self._loader_inst)
                self.n_batch = 1

            self.iter += 1
            loss = self.update(data, device)
            self.print(f'[iter_sum: {self.iter}][Iter: {i}/{T}][Batch: {self.n_batch}/{len(self.loader)}][Loss: {loss.item():.7f} (Avg: {self.train_meter.avg:.7f})]')

            self.loss_tracker.step(self.iter, loss)

            # Validation
            if (self.validation is not None) and (not no_val) and (i % self.cfg_train['cv_step']==0):
            # if (not no_val) and (self.iter % self.cfg_train.cv_step==0):
                patience_end = self.validate()
                if patience_end: # If patience has reached, stop training
                    self.print('Patience met, stopping training')
                    return None # return None since double break is impossible in python

    def update(self, data, device):
        '''
        This is where a lot of errors happen, so there's a pdb to save time.
        When there's error within a single update, use pdb to figure out the shape, device, tensor dtype, and more.
        '''
        try:
            loss = self.forward(data, device)

            self.op.zero_grad()
            loss.backward()
            self.op.step()
            return loss

        except Exception as e:
            log.warning(e)
            import pdb; pdb.set_trace()

    def forward(self, data, device):
        """
        Forward pass. Receive data and return the loss function.
        Inherit Node and define new forward() function to build custom forward pass.
        """

        if type(data)==tuple:
            x, y = data
        elif type(data)==dict:
            x, y = data['x'].to(device), data['y'].to(device)
        else:
            raise Exception(f'return type from dataset undefined in hideandseek.N.Node: {type(data)}')

        if self.amp:
            # Mixed precision for acceleration
            with torch.cuda.amp.autocast():
                # TODO: Modularize this part so that custom forward pass is possible.
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
        else:
            # TODO: Same as above
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

        N = len(y)
        self.train_meter.step(loss.item(), N)

        return loss

    def epoch_f(self):
        '''function to call every other epoch. May be used in subclass nodes'''
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
