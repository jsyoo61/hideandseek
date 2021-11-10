'''


'''
import logging
import os
import shutil
from copy import deepcopy as dcopy

import torch
import torch.optim as optim
import torch.utils.data as D

import tools # since T is used as a variable in this module, refrain from "import tools as T"
import tools.torch
import tools.modules

from . import validation as V

__all__ = [
'Node'
]

# %%
log = logging.getLogger(__name__)

# %%
class Node:
    '''Local Node for training'''
    def __init__(self, model, dataset, cv, cfg_train, criterion, MODEL_DIR, NODE_DIR, name='default', verbose=True, amp=False):
        '''
        :param model: torch.nn.Module object
        :param dataset: torch.utils.data.Dataset object
            This dataset is used for training.
            Recommended return format from dataset is: {'x': x, 'y': y}
            where 'x' is the input, and 'y' is the target values.



        :param cv: dict of Validation objects

        :param cfg_train: dict-like object which contains:
            lr
            batch_size
            weight_decay (optional)
            patience (optional)

        :param criterion: torch.nn.Module object, used to compute loss function

        Recommended way of making hs.node.Node object is like the following:

            kwargs = {'model': model, 'dataset': train_dataset, 'cv': None, 'cfg_train': cfg.train,
                    'criterion': criterion, 'MODEL_DIR': path['model'], 'NODE_DIR': path['node'], 'verbose': True, 'amp': True}
            node = hs.node.Node(**kwargs)
        '''
        self.model = model
        self.dataset = dataset
        self.cv = cv
        self.cfg_train = dcopy(cfg_train)
        self.criterion = criterion
        self.MODEL_DIR = MODEL_DIR
        self.NODE_DIR = NODE_DIR
        self.name = name
        self.verbose = verbose
        self.amp = amp

        if self.MODEL_DIR is not None:
            os.makedirs(self.MODEL_DIR, exist_ok=True)
        if self.NODE_DIR is not None:
            os.makedirs(self.NODE_DIR, exist_ok=True)

        self.set_misc()

        self.iter = 0
        self.train_meter = tools.modules.AverageMeter()
        self.set_cv(cv)
        self.loss_tracker = tools.modules.ValueTracker()

        self._targets_type = self.targets_type()

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
            if hasattr(self.dataset, 'get_f'):
                self.print('get_f found in loader.dataset')

                self.misc.get_f = self.dataset.get_f

    def set_cv(self, cv):
        self.cv = cv
        if self.cv is not None:
            self.cv_tracker = {cv_name:{s:tools.modules.ValueTracker() for s in cv.scorer.keys()} for cv_name, cv in self.cv.items()}

    def validate(self, prefix=''):
        patience_end = False
        if self.cv is not None:
            for cv_name, cv in self.cv.items():
                score, _patience_end = self._validation(cv)
                if _patience_end: # If patience_end occurs a single time, return True
                    patience_end=True
                self.print(f'{prefix}[cv_name: {cv_name}] Validation Score: {score}')
                V.track_score(self.cv_tracker[cv_name], score, x=self.iter, label=self.name)
        return patience_end

    def _validation(self, cv):
        patience_end = False
        if type(cv)==V.EarlyStopping:
            score, patience_end = cv.step(self, os.path.join(self.MODEL_DIR, f'model_{self.iter}.pt'))
            if patience_end:
                self.print('patience met, flushing cv_history')
                cv.cv_history.clear()
        else:
            # score = cv.step(self, os.path.join(self.MODEL_DIR, f'model_{self.iter}.pt'))
            score = cv.step(self)
        return score, patience_end

    def step(self, T=None, horizon='epoch', new_op=True, no_val=False):
        '''
        Trains the model with the specified duration.
        '''
        if T is None: # When no tools is
            T = self.cfg_train['epoch']

        self.print(f'[Node: {self.name}][step: {T} ({horizon})]')
        device = tools.torch.get_device(self.model)
        self.criterion = self.criterion.to(device)

        if 'cv_step' not in self.cfg_train:
            self.cfg_train.cv_step = len(self.loader)
        if self.cv is not None:
            self.cv.reset()
            self.validate(prefix=prefix)

        # Make new optimizer
        if new_op or not hasattr(self, 'op'):
            # Weight decay optional
            self.op = optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'], weight_decay=self.cfg_train['weight_decay']) if 'weight_decay' in self.cfg_train \
                        else optim.Adam(self.model.parameters(), lr=self.cfg_train['lr'])
        self.loss_tracker = tools.modules.ValueTracker()
        self.generate_loader()

        if horizon == 'epoch':
            self._step_epoch(T=T, no_val=no_val, device=device)
        elif horizon=='step':
            self._step_step(T=T, no_val=no_val, device=device)
        else:
            raise Exception(f'Invalid horizon type: {horizon}')

        self.criterion = self.criterion.cpu()
        return self.loss_tracker

    def generate_loader(self):
        if self.dataset is not None:
            # To avoid batch normalization layers from raising exceptions
            self.loader = D.DataLoader(self.dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=True) if len(self.dataset) % self.cfg_train['batch_size'] == 1 \
                    else D.DataLoader(self.dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=False)
        else:
            log.warning('No dataset found. Skipping generate_loader()')

    def update(self, data, device, prefix=''):
        try:
            x = data['x'].to(device)
            y = data['y'].to(device)
            N = len(y)

            if self.amp:
                # Mixed precision for acceleration
                with torch.cuda.amp.autocast():
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
            else:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

            self.op.zero_grad()
            loss.backward()
            self.op.step()
            self.train_meter.step(loss.item(), N)

            self.print(f'{prefix}[Loss: {loss.item():.7f} (Avg: {self.train_meter.avg:.7f})]')
            return loss.item()
        except Exception as e:
            log.warning(e)
            import pdb; pdb.set_trace()

    def _step_epoch(self, T, prefix='', no_val=False, device=None):
        device = device if device is not None else tools.torch.get_device(self.model)

        _iter = 0
        for epoch in range(1, T+1):
            self.train_meter.reset()
            self.epoch_f()
            # There may be one or more loaders, but self.loader is the standard of synchronization
            # Recommend modifying self.update() to use other loaders
            for batch_i, data in enumerate(self.loader, 1):
                self.iter += 1
                _iter += 1
                loss = self.update(data, device, prefix=f'{prefix}[iter_sum: {self.iter}][Epoch: {epoch}/{T}][Batch: {batch_i}/{len(self.loader)}]')
                self.loss_tracker.step(self.iter, loss)

                # Cross Validation
                if (self.cv is not None) and (not no_val) and (_iter % self.cfg_train.cv_step==0):
                    patience_end = self.validate(prefix=prefix)
                    # If patience has reached, stop training
                    if patience_end:
                        self.print('Patience met, stopping training')
                        return None

    def _step_step(self, T, prefix='', no_val=False, device=None):
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
            loss = self.update(data, device, prefix=f'{prefix}[iter_sum: {self.iter}][Iter: {i}/{T}][Batch: {self.n_batch}/{len(self.loader)}]')
            self.loss_tracker.step(self.iter, loss)

            # Cross Validation
            if (self.cv is not None) and (not no_val) and (i % self.cfg_train.cv_step==0):
            # if (not no_val) and (self.iter % self.cfg_train.cv_step==0):
                patience_end = self.validate(prefix=prefix)
                # If patience has reached, stop training
                if patience_end:
                    self.print('Patience met, stopping training')
                    return None

    def epoch_f(self):
        '''function to call every other epoch. May be used in subclass nodes'''
        pass

    def targets_type(self):
        if self.dataset is not None and hasattr(self.dataset, 'targets_type'):
            return self.dataset.targets_type
        elif self.cv is not None and 'val' in self.cv:
            if type(self.cv['val'].dataset)==list:
                return self.cv['val'].dataset[0].targets_type
            else:
                return self.cv['val'].dataset.targets_type
        else:
            return None

    def post_train(self, val_dataset=None):
        # Deafult to self.cv['val'].dataset
        if val_dataset is None:
            assert 'val' in self.cv, f'if no val_dataset is given, then "val" must exist in self.cv: {self.cv.keys()}'
            val_dataset = self.cv['val'].dataset

        if self._targets_type == 'binary':
            self.print(f'[Node: {self.name}][post_train] binary classfication: choose threshold based on validation set')
            self.threshold = E.binary_threshold(self, val_dataset)
        else:
            self.print(f'[Node: {self.name}][post_train] no post-train procedure for _targets_type: {self._targets_type}')

    '''
    Save/Load functions
    state_dict()
    load_state_dict()

    save()
    save_best()
    load()
    '''
    def state_dict(self):
        state_dict = {
        # 'cv_tracker': self.cv_tracker,
        'loss_tracker': self.loss_tracker,
        } # tools.TDict raises error when called in tools.save_pickle
        if 'get_f' in self.misc:
            state_dict['misc.get_f'] = self.misc.get_f
        if hasattr(self, 'threshold'):
            state_dict['threshold'] = self.threshold
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dcopy(state_dict)

        if 'misc.get_f' in state_dict.keys():
            self.print(f'Updating get_f: {state_dict["misc.get_f"]}')
            self.misc.get_f = state_dict['misc.get_f']
            if self.dataset is not None:
                self.print('updating get_f to loader...')
                self.dataset.get_f = self.misc.get_f
            del state_dict['misc.get_f']

        if 'threshold' in state_dict.keys():
            self.print(f'Updating threshold: {state_dict["threshold"]}')
            self.threshold = state_dict['threshold']
            del state_dict['threshold']

        self.__dict__.update(state_dict) # update attributes

    def save(self, path=None, best=True):
        '''
        Save the following:
        state_dict() -> path/state_dict.p
        model.state_dict() -> path/model.pt
        '''
        path = self.NODE_DIR if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[save] Saving node info to: {state_dict_path}')
        state_dict = self.state_dict()
        self.print(f'[save] state_dict: {list(state_dict.keys())}')
        tools.save_pickle(state_dict, state_dict_path)

        model_path = os.path.join(path, 'model.pt')
        if best:
            if 'val' not in self.cv:
                self.print(f'[save][best: {best}] "val" not in self.cv, Saving current model -> [{model_path}]')
                torch.save(self.model.state_dict(), model_path)
            else:
                if self.cv['val'].best_model != None:
                    self.print(f'[save][best: {best}] Saving [{self.cv["val"].best_model}] -> [{model_path}]')
                    shutil.copy(self.cv["val"].best_model, model_path)
                else:
                    self.print(f'[save][best: {best}] no best_model in self.cv["val"], Saving current model -> [{model_path}]')
                    torch.save(self.model.state_dict(), model_path)
        else:
            self.print(f'[save][best: {best}] Saving model to: {model_path}')
            torch.save(self.model.state_dict(), model_path)

    def load(self, path=None):
        '''
        load state_dict() and model:
        path/node.p -> state_dict
        path/model.pt -> model.state_dict
        '''
        path = self.NODE_DIR if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[load] Loading from path: {state_dict_path}')
        state_dict = tools.load_pickle(state_dict_path)
        self.print(f'[load] Updating: {list(state_dict.keys())}')
        self.load_state_dict(state_dict)

        model_path = os.path.join(path, 'model.pt')
        self.print(f'[load] Loading model from: {model_path}')
        self.model.load_state_dict(torch.load(model_path))
