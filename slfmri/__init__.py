from .lib import io
from .lib import volume
from .lib import utils
from .lib import signal
from .lib.io import Atlas

__version__ = '0.0.6'
__all__ = ['utils', 'signal', 'io', 'volume', 'Atlas']

load = io.load
