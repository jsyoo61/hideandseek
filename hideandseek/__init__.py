__version__ = "0.2.0" # Starting 0.2.0, somewhat functional
from . import eval as E
from .trainer import Trainer
from . import validation as V
from . import utils as U
from . import dataset as D
from .dataset import Dataset

# TODO: Change naming: eval -> evaluation (eval is a python function)
# TODO: Make scripts for projects to fork from
# TODO: Refine tutorial