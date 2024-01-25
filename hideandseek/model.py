import torch

import tools

from . import utils as U

class BaseModel(object):
    call_arguments = []

    def __init__(self, network, amp=False, misc={}):
        self.network = network
        self.amp = amp
        self.misc = misc # For certain metadata used for pre/postprocessing

    def __call__(self, batch=False, *args, **kwargs):
        """
        The preprocess/postprocess assumes batch dimension always exists.

        Parameters
        ----------
        batch : bool, default=False
            Whether the input has the batch dimension.
            If False, a batch dimension (axis=0) will be added
        *args, **kwargs : argument specific to each model

        Returns
        -------
        arrays : tuple of ndarrays
            The return value of a model
        """
        # Trim call_arguments (which are mostly not np.ndarray)
        call_arguments = {k:v for k, v in kwargs.items() if k in self.call_arguments}
        kwargs = {k:v for k, v in kwargs.items() if k not in self.call_arguments}

        # Setup inference
        self.network.eval()
        device = tools.torch.get_device(self.network)
        if not batch: # If the given values are does not have batch dimension,
            args = tuple(U.add_batch_dim(arg) for arg in args)
            kwargs = {k: U.add_batch_dim(v) for k, v in kwargs.items()}

        # Preprocess & Convert to torch.Tensor
        arrays = self.preprocess(*args, **kwargs)
        arrays = arrays if isinstance(arrays, tuple) else (arrays,)
        tensors = [torch.as_tensor(array, device=device) for array in arrays]

        # Forward pass without gradient & Convert to numpy.ndarray
        if self.amp:
            with torch.autocast(device_type=device.type):
                with torch.no_grad(): tensors = self.forward(*tensors, **call_arguments)
        else:
            with torch.no_grad(): tensors = self.forward(*tensors, **call_arguments)
        tensors = tensors if isinstance(tensors, tuple) else (tensors,)
        arrays = [tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]

        # Postprocess
        arrays = self.postprocess(*arrays)
        if not batch:
            arrays = tuple(U.remove_batch_dim(array) if isinstance(array, (np.ndarray, dict)) else array for array in arrays) if isinstance(arrays, tuple) else U.remove_batch_dim(arrays)

        return arrays # May be tuple or numpy.ndarray

    def forward(self, *args, **kwargs):
        '''
        May be overridden in children classes
        '''
        return self.network(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        '''
        Preprocess the arguments into a form that network could compute.
        '''
        return *args, *kwargs.values()

    def inv_preprocess(self, *args, **kwargs):
        return *args, *kwargs.values()

    def postprocess(self, *args, **kwargs):
        '''
        Preprocess the arguments into
        '''
        return *args, *kwargs.values()

    def inv_postprocess(self, *args, **kwargs):
        return *args, *kwargs.values()
