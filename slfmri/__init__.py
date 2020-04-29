from .tools import *
from .norm import *
from .qc import *
from .filter import *
from .rsparam import *
from .io import *
from .lib.tmpobj import Atlas
__version__ = '0.0.1'
__all__ = ['get_funcobj', 'apply_funcobj', 'isfuncobj', 'get_volreg', 'iszero', 'get_filepath', 'mkdir', # tools module
           'demean', 'modenorm', 'standardization', # norm module
           'tsnr', 'bold_meanstd', 'dvars', 'mparam_ard', 'mparam_atd', 'mparam_fd', # qc module
           'bandpass', 'estimate_pc', 'linear_regression', 'nuisance_regression', 'polynomial_fit', # filters
           'alff', 'kandallw', 'periodogram', 'reho', 'welch', # rsparam module
           'load', 'save_to_nii', # io module
           'Atlas'
           ]