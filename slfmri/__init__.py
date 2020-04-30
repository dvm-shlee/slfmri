from .lib.utils import *
from slfmri.lib.signal.norm import *
from slfmri.lib.analysis.qc import *
from slfmri.lib.signal.filter import *
from slfmri.lib.analysis.rs import *
from .lib.io import *
from .deprecated.tmpobj import Atlas
__version__ = '0.0.1'
__all__ = [
    'get_funcobj', 'apply_funcobj', 'isfuncobj', 'get_volreg', 'iszero', 'get_filepath', 'mkdir', # tools module
    'demean', 'modenorm', 'standardization', # norm module
    'tsnr', 'bold_meanstd', 'dvars', 'mparam_ard', 'mparam_atd', 'mparam_fd', # qc module
    'bandpass', 'estimate_pc', 'linear_regression', 'nuisance_regression', 'polynomial_fit', # filters
    'alff', 'kandallw', 'periodogram', 'reho', 'welch', # rsparam module
    'load', 'save_to_nii', # io module
    'Atlas'
]