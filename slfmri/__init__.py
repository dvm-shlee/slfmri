from .lib import *
from .ui import *
from .lib.io import Atlas

__version__ = '0.1.0'
__all__ = ['utils', 'filter', 'io', 'metrics', 'stats', 'plot', 'Atlas', 'ui']
load = io.load
PathMan = io.PathMan
