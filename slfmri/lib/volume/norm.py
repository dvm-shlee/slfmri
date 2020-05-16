from ..errors import *
from ..timeseries import calc_modenorm
from ..utils import get_funcobj, apply_funcobj


# space normalization tools
def modenorm(func_img, mask_img=None, mode=1000, io_handler=None):
    """
    Mode normalization

    :param func_img: time series data
    :param mask_img:
    :param mode: target mean value
    :return: processed_img
    """
    if mask_img is None:
        indices = np.nonzero(func_img)
    else:
        indices = np.nonzero(mask_img)
    mean = func_img[indices].mean().copy()
    modenorm_obj = get_funcobj(calc_modenorm, mean, mode)
    return apply_funcobj(modenorm_obj, func_img, mask_img, io_handler)