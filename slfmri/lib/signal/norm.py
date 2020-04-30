# functions contains space normalization


# signal normalization tools
def demean(signal, axis=0):
    return signal - signal.mean(axis)


def standardization(signal, axis=0):
    demeaned = demean(signal, axis=axis)
    
    if demeaned.std(axis) is 0:
        return demeaned * 0
    else:
        norm_data = demeaned / demeaned.std(axis)
    return norm_data


def calc_modenorm(signal, mean, mode):
    return (signal - mean) * mode / mean + mode


# space normalization tools
def modenorm(func_img, mask_img=None, mode=1000):
    """
    Mode normalization

    :param func_img: time series data
    :param mode: target mean value
    :return: processed_img
    """
    import numpy as np
    from ..utils import get_funcobj, apply_funcobj
    if mask_img is None:
        indice = np.nonzero(func_img)
    else:
        indice = np.nonzero(mask_img)
    mean = func_img[indice].mean().copy()
    mnorm_obj = get_funcobj(calc_modenorm, mean, mode)
    return apply_funcobj(mnorm_obj, func_img, mask_img)