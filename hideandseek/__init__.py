__version__ = "0.1.0"
from . import eval as E
from . import node as N
from . import validation as V

# TODO: Change V.EarlyStopping funcionality into node. patience & primary scorer etc should be managed by node, not validation.
# TODO: Change naming: cross validation -> validation
