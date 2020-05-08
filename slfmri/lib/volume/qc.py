from ..errors import *
from ..utils import get_funcobj, apply_funcobj
from ..timeseries import tsnr, demean


def dvars(func_img, mask_img=None):
    if mask_img is not None:
        indices = np.nonzero(mask_img)
    else:
        indices = np.nonzero(func_img.mean(-1))
    diff_img = np.diff(func_img[indices], axis=-1)
    dvars_ = np.sqrt(np.square(diff_img).mean(0))
    return np.insert(dvars_, 0, np.nan)


def bold_mean_std(func_img, mask_img=None):
    if mask_img is not None:
        indices = np.nonzero(mask_img)
    else:
        indices = np.nonzero(func_img.mean(-1))
    demean_obj = get_funcobj(demean, axis=-1)
    demeaned_img = apply_funcobj(demean_obj, func_img, mask_img)

    return demeaned_img[indices].mean(0), demeaned_img[indices].std(0)


def img_tsnr(func_img, mask_img=None):
    tsnr_obj = get_funcobj(tsnr)
    tsnr_img = apply_funcobj(tsnr_obj, func_img, mask_img)
    return tsnr_img
