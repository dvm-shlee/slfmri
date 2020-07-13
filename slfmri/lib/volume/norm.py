from ..errors import *
from ..signal import modenorm, standardize
from ..utils import get_funcobj, apply_funcobj


# space normalization tools
def vol_modenorm(func_img, mask_img=None, mode=1000, io_handler=None):
    """ Mode normalization

    Args:
        func_img: time series data
        mask_img: brain mask
        mode: target mean value
        io_handler: file object
    Returns:
        normalized image
    """
    if mask_img is None:
        indices = np.nonzero(func_img)
    else:
        indices = np.nonzero(mask_img)
    mean = func_img[indices].mean().copy()
    modenorm_obj = get_funcobj(modenorm, mean, mode)
    return apply_funcobj(modenorm_obj, func_img, mask_img, io_handler)


def vol_standardize(func_img, mask_img=None, io_handler=None):
    """
    Standard normalization

    Args:
        func_img: time series data
        mask_img: brain mask
        io_handler: file object
    Returns:
        normalized image
    """
    standardized_obj = get_funcobj(standardize)
    return apply_funcobj(standardized_obj, func_img, mask_img, io_handler)