from .lib import io
from .lib import volume
from .lib import utils
from .lib import timeseries
from .deprecated.tmpobj import Atlas

__version__ = '0.0.3'
__all__ = ['utils', 'timeseries', 'io', 'volume', 'Atlas']

load = io.load
