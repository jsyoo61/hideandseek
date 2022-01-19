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
from . import utils as U

__all__ = [
'Node'
]

# %%
log = logging.getLogger(__name__)

# %%
class Node:
    '''Local Node for training'''
    def __init__(self, model, dataset, cfg_train, criterion, model_dir=None, node_dir=None, validation=None, earlystopper=None, name='default', verbose=True, amp=False):
        '''
        :param model: torch.nn.Module object
        :param dataset: torch.utils.data.Dataset object
            This dataset is used for training.
            Recommended return format from dataset is: {'x': x, 'y': y}
            where 'x' is the input, and 'y' is the target values.



        :param validation: dict of Validation objects

        :param cfg_train: dict-like object which contains:
            lr
            batch_size
            weight_decay (optional)
            patience (optional)

        :param criterion: torch.nn.Module object, used to compute loss function

        Recommended way of making hs.node.Node object is like the following:

            kwargs = {'model': model, 'dataset': train_dataset, 'validation': None, 'cfg_train': cfg.train,
                    'criterion': criterion, 'model_dir': path['model'], 'node_dir': path['node'], 'verbose': True, 'amp': True}
            node = hs.node.Node(**kwargs)
        '''
        # Store configurations
        self.model = model
        self.dataset = dataset
        self.cfg_train = dcopy(dict(cfg_train))
        self.criterion = criterion
        self.model_dir = model_dir
        self.node_dir = node_dir
        self.validation = validation if issubclass(type(validation), dict) else V.VDict({'default': validation}) # wrap with VDict if single validation object is given.
        self.earlystopper = earlystopper
        self.name = name
        self.verbose = verbose
        self.amp = amp

        # Initializations
        if self.model_dir is None:
            self.print('model_dir not specified, default to "model"')
            self.model_dir = 'model'
        if os.path.exists(self.model_dir): log.warning(f'path: {self.model_dir} exists. Be careful')
        os.makedirs(self.model_dir, exist_ok=True)

        if self.node_dir is None:
            self.print('node_dir not specified, default to "node"')
            self.node_dir = 'node'
        if os.path.exists(self.node_dir): log.warning(f'path: {self.node_dir} exists. Be careful')
        os.makedirs(self.node_dir, exist_ok=True)

        if self.validation is None and self.earlystopper is not None: log.warning('validation is None but earlystopper is given. earlystopper will be ignored')
        if self.earlystopper is not None:
            assert earlystopper.target_validation in self.validation.keys(), 'earlystopper.target_valiation not provided in validation'
            assert earlystopper

        self.train_meter = tools.modules.AverageMeter()
        self.loss_tracker = tools.modules.ValueTracker()
        self.set_misc()
        self.reset()

        self._targets_type = self.infer_targets_type()

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

    # def step(self, T=None, horizon='epoch', new_op=True, no_val=False):
    def train(self, epoch=None, step=None, new_op=True, no_val=False):
        '''
        Trains the model with the specified duration.
        '''
        assert epoch is None or step is None, f'only one of epoch or step  can be specified. received epoch: {epoch}, step: {step}'
        if step is None:
            horizon = 'epoch'
            if epoch is None: # When neither epoch or step are specified
                assert 'epoch' in self.cfg_train, 'key "epoch" must be provided in cfg_train when argument "T" is not specified.'
                T = self.cfg_train['epoch']
            else:
                T = epoch
        else:
            assert epoch is None, f'only one of epoch or step can be specified. received epoch: {epoch}, step: {step}'
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
        '''
        inherit Node and define new forward() function to build custom forward pass
        '''
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

    def infer_targets_type(self):
        if self.dataset is not None and hasattr(self.dataset, 'targets_type'):
            return self.dataset.targets_type
        elif self.validation is not None and 'val' in self.validation:
            if type(self.validation['val'].dataset)==list:
                return self.validation['val'].dataset[0].targets_type
            else:
                return self.validation['val'].dataset.targets_type
        else:
            return None

    def post_train(self, val_dataset=None):
        # Deafult to self.validation['val'].dataset
        if val_dataset is None:
            assert 'val' in self.validation, f'if no val_dataset is given, then "val" must exist in self.validation: {self.validation.keys()}'
            val_dataset = self.validation['val'].dataset

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
        path = self.node_dir if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[save] Saving node info to: {state_dict_path}')
        state_dict = self.state_dict()
        self.print(f'[save] state_dict: {list(state_dict.keys())}')
        tools.save_pickle(state_dict, state_dict_path)

        model_path = os.path.join(path, 'model.pt')
        if best:
            if self.earlystopper is None:
                self.print(f'[save][best: {best}] earlystopper not defined. Saving current model -> [{model_path}]')
                torch.save(self.model.state_dict(), model_path)
            else:
                if self.earlystopper.best_model != None:
                    self.print(f'[save][best: {best}] Saving [{self.earlystopper.best_model}] -> [{model_path}]')
                    shutil.copy(self.earlystopper.best_model, model_path)
                else:
                    self.print(f'[save][best: {best}] no best_model in self.earlystopper, Saving current model -> [{model_path}]')
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
        path = self.node_dir if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[load] Loading from path: {state_dict_path}')
        state_dict = tools.load_pickle(state_dict_path)
        self.print(f'[load] Updating: {list(state_dict.keys())}')
        self.load_state_dict(state_dict)

        model_path = os.path.join(path, 'model.pt')
        self.print(f'[load] Loading model from: {model_path}')
        self.model.load_state_dict(torch.load(model_path))
