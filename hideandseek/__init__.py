__version__ = "0.3.4" # 0.3.0 Now, Trainer.forward() should return the loss not outputs.
from . import evaluation as E
from .trainer import Trainer
# from . import validation as V
from . import utils as U
from . import dataset as D
from . import fv
from .dataset import Dataset

# TODO: Make scripts for projects to fork from
# TODO: Refine tutorial