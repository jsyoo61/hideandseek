import torch

class Dataset(torch.utils.data.Dataset):
    '''
    inherit this class and update: __init__, get_x, get_y, __len__
    '''
    def __repr__(self):
        data_info = {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in self.__dict__.items() if not k.startswith('_')}
        return f'<{self.__class__.__name__} object> with: \n{data_info}'

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self):
            data = self[self._i]
            self._i +=1
            return data
        else:
            raise StopIteration

    def __getitem__(self, idx):
        x, y = self.get_x(idx), self.get_y(idx)

        return {'x': x, 'y': y}

    def __len__(self):
        pass # Need to define

    def get_x(self, idx):
        # Should return x
        pass
    
    def get_y(self, idx):
        # Should return y
        pass

    def get_y_all(self):
        # Optional for fast computation when getting loss weights. return all y
        pass

class SimpleDataset(Dataset):
    def __init__(self, x, y, convert_tensor=False):
        """
        Assume x, y are given in numpy arrays, although torch tensor might be compatible

        Parameters
        ----------
        x: array-like
        y: array-like
        convert_tensor: bool, default=False
            If True, convert and store x, y into torch tensors. This may require more memory.
        """
        self.x = torch.as_tensor(x) if convert_tensor else x
        self.y = torch.as_tensor(y) if convert_tensor else y
        self._convert_tensor = convert_tensor

    def get_x(self, idx):
        return self.x[idx] if self._convert_tensor else torch.as_tensor(self.x[idx])

    def get_y(self, idx):
        return self.y[idx] if self._convert_tensor else torch.as_tensor(self.y[idx])

    def get_y_all(self):
        return self.y if self._convert_tensor else torch.as_tensor(self.y)

    def __len__(self):
        return len(self.y)