from ..errors import *
from ..utils import get_funcobj, apply_funcobj
from ..signal import tsnr, demean


def dvars(func_img, mask_img=None):
    if mask_img is not None:
        indices = np.nonzero(mask_img)
    else:
        indices = np.nonzero(func_img.mean(-1))
    diff_img = np.diff(func_img[indices], axis=-1)
    dvars_ = np.sqrt(np.square(diff_img).mean(0))
    return np.insert(dvars_, 0, np.nan)


def bold_mean_std(func_img, mask_img=None, io_handler=None):
    if mask_img is not None:
        indices = np.nonzero(mask_img)
    else:
        indices = np.nonzero(func_img.mean(-1))
    demean_obj = get_funcobj(demean, axis=-1)
    demeaned_img = apply_funcobj(demean_obj, func_img, mask_img, io_handler)

    return demeaned_img[indices].mean(0), demeaned_img[indices].std(0)


def vol_tsnr(func_img, mask_img=None, io_handler=None):
    tsnr_obj = get_funcobj(tsnr)
    tsnr_img = apply_funcobj(tsnr_obj, func_img, mask_img, io_handler)
    return tsnr_img

# def img_tsnr(func_img, mask_img=None, io_handler=None):
#     if mask_img is None:
#         mask_img = np.abs(func_img.mean(-1))
#     mean_data = func_img.mean(-1)[mask_img > 0]
#     std_data = func_img.std(-1)[mask_img > 0]
#     tsnr_img = np.zeros(mask_img.shape)
#     tsnr_img[mask_img > 0] = mean_data / std_data
#     return tsnr_img
