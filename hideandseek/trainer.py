'''


'''
import logging
import os
import shutil
from copy import deepcopy as dcopy

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as D

import tools # since T is used as a variable in this module, refrain from "import tools as T"
import tools.torch
import tools.modules

from . import utils as U
from . import evaluation as E
from . import model as M

# __all__ = [
# 'Trainer'
# ]

# %%
log = logging.getLogger(__name__)

# %%
class Trainer:
    def __init__(self, network, train_dataset=None, cfg_train={}, criterion=None, network_dir=None, model_dir=None, cfg_val=None, val_dataset=None, val_metrics=None, name='default', verbose=True, amp=False, reproduce=True):
        '''
        Things Trainer do:
        - train network with early stopping, given hyperparameters
        - save/load network with necessary preprocessing pipeline

        :param network: torch.nn.Module object
        :param train_dataset: torch.utils.data.Dataset object
            This dataset is used for training.
            Recommended return format from dataset is: {'x': x, 'y': y}
            where 'x' is the input, and 'y' is the target values.

        :param validation: dict of Validation objects

        :param cfg_train: dict-like object which contains:
            lr
            batch_size
            weight_decay (optional)
            cv_step (optional)
        
        :param cfg_val (optional): dict-like object which contains:
            increase_better
            patience
            batch_size
            target_dataset (optional, if val_dataset is dict for multiple datasets)
            criterion (optional, if val_metrics is dict for multiple metrics) 

        :param criterion: torch.nn.Module object, used to compute loss function

        Recommended way of making hs.node.Node object is like the following:

            kwargs = {'network': Network(), 'train_dataset': train_dataset, 'validation': None, 'cfg_train': cfg.train,
                    'criterion': criterion, 'network_dir': path['network'], 'model_dir': path['node'], 'verbose': True, 'amp': True}
            node = hs.node.Node(**kwargs)
        '''
        # Store configurations
        self.network = network
        self.train_dataset = train_dataset
        self.cfg_train = dcopy(dict(cfg_train))
        self.criterion = criterion
        self.network_dir = network_dir
        self.model_dir = model_dir

        self.cfg_val = dcopy(dict(cfg_val)) if cfg_val is not None else {}
        self.val_dataset = val_dataset
        self.val_metrics = val_metrics
        self.earlystopper = EarlyStopper(increase_better=cfg_val['increase_better'], patience=cfg_val['patience']) if self.val_dataset is not None else None
        if 'target_dataset' in self.cfg_val:
            if type(self.val_dataset)==dict:
                assert self.cfg_val['target_dataset'] in self.val_dataset, f'[Validation setup] target_dataset: {self.cfg_val["target_dataset"]} not found in val_dataset: {self.val_dataset.keys()}'
            assert ~((self.val_dataset is not None) ^ (self.val_metrics is not None)), f'val_dataset and val_metrics must be given together or neither. val_dataset: {self.val_dataset}, val_metrics: {self.val_metrics}'

        self.name = name
        self.verbose = verbose
        self.amp = amp
        self.reproduce = reproduce

        # Initializations
        if self.network_dir is not None:
            if os.path.exists(self.network_dir): log.warning(f'path: {self.network_dir} exists. Be careful')
            os.makedirs(self.network_dir, exist_ok=True)

        if self.model_dir is not None:
            if os.path.exists(self.model_dir): log.warning(f'path: {self.model_dir} exists. Be careful')
            os.makedirs(self.model_dir, exist_ok=True)

        self.train_meter = tools.modules.AverageMeter() # Tracks loss per epoch
        self.loss_tracker = tools.modules.ValueTracker() # Tracks loss over all training
        self.set_model()
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

    def set_model(self):
        '''Set miscellaneous variables'''
        self.model = M.BaseModel(self.network, amp=self.amp)
        self.misc = tools.TDict()
        if self.train_dataset is not None:
            if hasattr(self.train_dataset, 'get_f'): # get_f equal to preprocessing pipeline before feeding to network
                self.print('get_f found in loader.dataset')
                self.misc.get_f = self.train_dataset.get_f

    def validate(self):
        patience_end = False
        self.print(f'[Node: {self.name}] Validation')
        if self.val_dataset is not None:
            if type(self.val_dataset)==dict:
                scores = {}
                for name, dataset in self.val_dataset.items():
                    d_results = E.test_model(self.model, dataset, batch_size=self.cfg_val['batch_size'], amp=self.amp)
                    if any([np.isnan(v).any() for v in d_results.values()]):
                        log.warning(f'[Node: {self.name}][validate] NaN found in d_results. Skipping validation...')
                        score = {self.cfg_val['criterion']: self.earlystopper.best_score}
                    else:
                        score = E.evaluate(d_results, self.val_metrics)
                    scores[name] = score
                    self.print(f'[dataset_name: {name}] Validation Score: {score}')
                score = scores[self.cfg_val['target_dataset']]
            else:
                d_results = E.test_model(self.model, self.val_dataset, batch_size=self.cfg_val['batch_size'], amp=self.amp)
                if any([np.isnan(v).any() for v in d_results.values()]):
                    log.warning(f'[Node: {self.name}][validate] NaN found in d_results. Skipping validation...')
                    score = {self.cfg_val['criterion']: self.earlystopper.best_score}
                else:
                    score = E.evaluate(d_results, self.val_metrics)
                self.print(f'Validation Score: {score}')
            
            if self.earlystopper is not None:
                if type(score) is dict:
                    score_ = score[self.cfg_val['criterion']]
                else:
                    score_ = score
                    
                patience_end = self.earlystopper.step(self, score_)
                if patience_end:
                    self.print('patience met, flushing earlystopper history')
                    self.earlystopper.history.reset()
                    
        torch.cuda.empty_cache()
        return patience_end

    def generate_loader(self):
        if self.train_dataset is not None:
            reproduce_kwargs = U.reproducible_worker_dict() if self.reproduce else {}
            drop_last = len(self.train_dataset) % self.cfg_train['batch_size'] == 1 # To avoid batch normalization layers from raising exceptions
            self.loader = D.DataLoader(self.train_dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=drop_last, **reproduce_kwargs)
            # self.loader = D.DataLoader(self.train_dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=True, **U.reproducible_worker_dict()) if len(self.train_dataset) % self.cfg_train['batch_size'] == 1 \
            #         else D.DataLoader(self.train_dataset, batch_size=self.cfg_train['batch_size'], shuffle=True, drop_last=False, **U.reproducible_worker_dict())
        else:
            log.warning('No train_dataset found. Skipping generate_loader()')

    def train(self, epoch=None, new_op=True, no_val=False, step=None, reset_loss_tracker=False):
        '''
        Trains the network with the specified duration.
        '''
        assert epoch is None or step is None, f'only one of epoch or step can be specified. received epoch: {epoch}, step: {step}'
        if step is None:
            horizon = 'epoch'
            if epoch is None: # When neither epoch or step are specified
                assert 'epoch' in self.cfg_train, 'key "epoch" must be provided in cfg_train, or the argument "epoch" must be provided \
                                                when argument "step" is not specified.'
                epoch = self.cfg_train['epoch']
            log.debug(f'[Node: {self.name}] train for {epoch} epochs')
        else:
            assert epoch is None, f'Either epoch or step must be specified. Received epoch: {epoch}, step: {step}'
            horizon = 'step'
            step = step
            log.debug(f'[Node: {self.name}] train for {step} steps')

        self.network.train()
        device = tools.torch.get_device(self.network)
        self.criterion = self.criterion.to(device)
        if reset_loss_tracker: self.loss_tracker.reset()
        self.generate_loader()
        '''
        There may be one or more loaders, but self.loader is the standard of synchronization
        Either return multiple values from dataset, or modify self.forward to use other loaders
        '''

        if self.val_dataset is not None:
            if 'cv_step' not in self.cfg_train:
                self.print('Node.validation given, but "cv_step" not specified in cfg_train. Defaults to 1 epoch')
                self.cfg_train['cv_step'] = len(self.loader)
            self.validate() # initial testing

        # Make new optimizer
        if new_op or not hasattr(self, 'op'):
            if not hasattr(self, 'op') and not new_op:
                log.warning("new_op=False when there's no pre-existing optimizer. Ignoring new_op...")
            
            l_kwargs = ['lr', 'weight_decay', 'amsgrad']
            kwargs = {key: self.cfg_train[key] for key in l_kwargs if key in self.cfg_train}
            self.op = optim.Adam(self.network.parameters(), **kwargs)

        if horizon == 'epoch':
            self._update_epoch(T=epoch, no_val=no_val, device=device)
        elif horizon=='step':
            self._update_step(T=step, no_val=no_val, device=device)

        # TODO: Return criterion back to its original device, meaning we have to store its previous device info
        self.criterion = self.criterion.cpu()
        return self.loss_tracker

    def _update_epoch(self, T, no_val=False, device=None):
        self._device = device if device is not None else tools.torch.get_device(self.network)

        _iter = 0
        for epoch in range(1, T+1):
            self.train_meter.reset()
            self.epoch_f()
            for batch_i, data in enumerate(self.loader, 1):
                self.iter += 1
                _iter += 1
                loss = self._update(data)
                self.print(f'[Node: {self.name}][iter_sum: {self.iter}][Epoch: {epoch}/{T}][Batch: {batch_i}/{len(self.loader)}][Loss: {loss:.7f} (Avg: {self.train_meter.avg:.7f})]')
                self.loss_tracker.step(self.iter, loss)

                # Validation
                if (self.val_dataset is not None) and (not no_val) and (_iter % self.cfg_train['cv_step']==0):
                    patience_end = self.validate()
                    if patience_end: # If patience has reached, stop training
                        self.print('Patience met, stopping training')
                        return None # return None since double break is impossible in python

    def _update_step(self, T, no_val=False, device=None):
        self._device = device if device is not None else tools.torch.get_device(self.network)
        if hasattr(self, '_loader_inst'): del self._loader_inst # Load data from the 1st batch

        for _iter in range(1, T+1):
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
            loss = self._update(data)
            self.print(f'[iter_sum: {self.iter}][Iter: {_iter}/{T}][Batch: {self.n_batch}/{len(self.loader)}][Loss: {loss:.6f} (Avg: {self.train_meter.avg:.6f})]')

            self.loss_tracker.step(self.iter, loss)

            # Validation
            if (self.val_dataset is not None) and (not no_val) and (_iter % self.cfg_train['cv_step']==0):
            # if (not no_val) and (self.iter % self.cfg_train.cv_step==0):
                patience_end = self.validate()
                if patience_end: # If patience has reached, stop training
                    self.print('Patience met, stopping training')
                    return None # return None since double break is impossible in python

    def _update(self, data):
        '''
        Pseudo function to support amp (automatic mixed precision)
        '''
        if self.amp:
            # Mixed precision for acceleration
            with torch.autocast(device_type=self._device.type):
                return self.update(data)
        else:
            return self.update(data)

    def update(self, data):
        """
        - Perform single update (forward/backward pass + gradient descent step) with the given data.
        - Store loss in self.train_meter
        - This is where a lot of errors happen, so there's a pdb to save time.
          When there's error, use pdb to figure out the shape, device, tensor dtype, and more.

        Parameters
        ----------
        data : tuple, list, or dict of tensors (Batch of data)
            This is received from a torch.utils.Data.DataLoader
            Depending on the given format, the data is fed to the forwad pass

        Returns
        -------
        loss : float
        """
        try:
            outputs, N = self._forward(data)
            loss = self._criterion(outputs)

            self.op.zero_grad()
            loss.backward()
            self.op.step()

            loss = loss.item()
            self.train_meter.step(loss, N)
            return loss

        except Exception as e:
            log.warning(e)
            import pdb; pdb.set_trace()

    def _forward(self, data):
        '''
        Pseudo function to support passing tuple or dict from a batch from dataloader to forward()
        '''
        datatype = type(data)
        # When data is given as a tuple/list
        if datatype is tuple or datatype is list:
            data = [x.to(self._device) for x in data]
            N = len(data[0]) # number of data in batch
            outputs = self.forward(*data)

        # When data is given as a dict
        elif datatype is dict:
            data = {key: value.to(self._device) for key, value in data.items()}
            N = len(next(iter(data.values()))) # number of data in batch
            outputs = self.forward(**data)

        else:
            raise Exception(f'return type from dataset must be one of [tuple, list, dict], received: {datatype}')
        return outputs, N

    def forward(self, x, y):
        """
        Forward pass: Receive data and pass values to criterion.
        Define forward for custom trainers.

        May return tuple or dictionary, whichever will be feeded to criterion.

        Defaults to args[0] feeding to self.network (assuming x), and args[1] feeding to self.criterion (assuming y)
        """
        
        y_hat = self.network(x)

        return y_hat, y

    def _criterion(self, outputs):
        outputstype = type(outputs)
        if outputstype is tuple or outputstype is list:
            loss = self.criterion(*outputs)
        elif outputstype==dict:
            loss = self.criterion(**outputs)
        elif outputstype==torch.Tensor:
            loss = self.criterion(outputs)
        else:
            raise Exception(f'return type from forward must be one of [tuple, list, dict, torch.Tensor], received: {type(outputs)}')
        return loss

    def epoch_f(self):
        # TODO: rename to forward hook? epoch_hook?
        '''function to call every other epoch. May be used in child class'''
        pass

    def infer_targets_type(self):
        if self.train_dataset is not None and hasattr(self.train_dataset, 'targets_type'):
            return self.train_dataset.targets_type
        elif self.val_dataset is not None:
            if type(self.val_dataset)==dict:
                for name, dataset in self.val_dataset.items():
                    if hasattr(dataset, 'targets_type'):
                        return dataset.targets_type
            elif hasattr(self.val_dataset, 'targets_type'):
                return self.val_dataset.targets_type
        else:
            return None

    def post_train(self, val_dataset=None):
        if val_dataset is None:
            assert self.val_dataset is not None, f'val_dataset must be given if self.val_dataset is None'
            if type(self.val_dataset)==dict:
                val_dataset = self.val_dataset[self.cfg_val['target_dataset']]
            else:
                val_datset = self.val_dataset

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
        'loss_tracker': self.loss_tracker,
        }
        if 'get_f' in self.misc:
            state_dict['misc.get_f'] = self.misc.get_f
        if hasattr(self, 'threshold'):
            state_dict['threshold'] = self.threshold
        if self.earlystopper is not None:
            state_dict['earlystopper.history'] = self.earlystopper.history
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dcopy(state_dict)

        if 'misc.get_f' in state_dict.keys():
            self.print(f'Updating get_f: {state_dict["misc.get_f"]}')
            self.misc.get_f = state_dict['misc.get_f']
            if self.train_dataset is not None:
                self.print('updating get_f to loader...')
                self.train_dataset.get_f = self.misc.get_f
            del state_dict['misc.get_f']

        if 'threshold' in state_dict.keys():
            self.print(f'Updating threshold: {state_dict["threshold"]}')
            self.threshold = state_dict['threshold']
            del state_dict['threshold']

        self.__dict__.update(state_dict) # update attributes
        
    def load_best_model(self):
        if self.earlystopper is not None:
            if self.earlystopper.best_network != None:
                self.print(f'Loading best network: {self.earlystopper.best_network}')
                self.network.load_state_dict(torch.load(self.earlystopper.best_network))
            else:
                self.print(f'No best_network found in self.earlystopper. No network loaded.')
        else:
            self.print('self.earlystopper not defined. No network loaded.')

    def save(self, path=None, best=True):
        '''
        Save the following:
        state_dict() -> path/state_dict.p
        network.state_dict() -> path/network.pt
        '''
        path = self.network_dir if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[save] Saving node info to: {state_dict_path}')
        state_dict = self.state_dict()
        self.print(f'[save] state_dict: {list(state_dict.keys())}')
        tools.save_pickle(state_dict, state_dict_path)

        network_path = os.path.join(path, 'network.pt')
        if best:
            if self.earlystopper is None:
                self.print(f'[save][best: {best}] earlystopper not defined. Saving current network -> [{network_path}]')
                torch.save(self.network.state_dict(), network_path)
            else:
                if self.earlystopper.best_network != None:
                    self.print(f'[save][best: {best}] Saving [{self.earlystopper.best_network}] -> [{network_path}]')
                    shutil.copy(self.earlystopper.best_network, network_path)
                else:
                    self.print(f'[save][best: {best}] no best_network in self.earlystopper, Saving current network -> [{network_path}]')
                    torch.save(self.network.state_dict(), network_path)
        else:
            self.print(f'[save][best: {best}] Saving network to: {network_path}')
            torch.save(self.network.state_dict(), network_path)

    def load(self, path=None):
        '''
        load state_dict() and network:
        path/node.p -> state_dict
        path/network.pt -> network.state_dict
        '''
        path = self.network_dir if path is None else path
        self.print(f'[Node: {self.name}]')

        state_dict_path = os.path.join(path, 'node.p')
        self.print(f'[load] Loading from path: {state_dict_path}')
        state_dict = tools.load_pickle(state_dict_path)
        self.print(f'[load] Updating: {list(state_dict.keys())}')
        self.load_state_dict(state_dict)

        network_path = os.path.join(path, 'network.pt')
        self.print(f'[load] Loading network from: {network_path}')
        self.network.load_state_dict(torch.load(network_path))

class EarlyStopper():
    """
    Returns True if the metric has stopped improving for the given number of patience steps.
    Saves network in the given path and remembers which network is the best.

    Parameters
    ----------
    arg : array-like of shape (n_samples,), default=None
        Argument explanation.
        If ``None`` is given, those that appear at least once
        .. versionadded:: 0.18
    
    Functions
    -------
    step()
        returns True/False
    """

    def __repr__(self):
        return f'<EarlyStopper>\npatience: {self.patience}\nincrease_better: {self.increase_better}'

    def __init__(self, increase_better, patience, save_dir='network_temp', discard_best=True):
        self.increase_better = increase_better
        self.patience = patience
        self.save_dir = save_dir
        self.discard_best = discard_best

        os.makedirs(self.save_dir, exist_ok=True)
        self.history = tools.modules.ValueTracker() # For now, only track single score
        self.reset()

    def step(self, trainer, score, filename=None):
        filename = f'network_{trainer.iter}.pt' if filename is None else filename

        self.history.step(trainer.iter, score)

        # Save best network
        if isbetter(score_best=self.best_score, score=score, increase_better=self.increase_better):
            path = os.path.join(self.save_dir, filename)

            log.info(f'[EarlyStopping] New best score: {self.best_score} -> {score}')
            log.info(f'[EarlyStopping] Saved network: {path}')
            torch.save(trainer.network.state_dict(), path)
            if self.discard_best and self.best_network!=None:
                os.remove(self.best_network)

            self.best_score = score
            self.best_network = path

            # Clear patience tracking
            self.history.reset()

        # patience_end == True when best network did not appear for self.patience times
        patience_end = self.patience == len(self.history)
        log.info(f'[EarlyStopping][patience: {len(self.history)}/{self.patience}]')

        return patience_end

    def reset(self):
        # log.info('[EarlyStopper] Resetting...')
        self.history.reset()
        self.best_network = None
        if self.increase_better:
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')

def isbetter(score_best: float, score: float, increase_better: bool):
    return (not increase_better and score<score_best) or (increase_better and score>score_best)
