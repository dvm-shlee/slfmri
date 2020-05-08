from ..errors import *
from ..timeseries.corr import kandallw


def reho(func_img, mask_img=None, nn=3):
    if mask_img is not None:
        indices = np.transpose(np.nonzero(mask_img))
    else:
        indices = np.transpose(np.nonzero(func_img.mean(-1)))
    reho_img = np.zeros(func_img.shape[:3])
    for i, t, k in indices:
        reho_img[i, t, k] = kandallw(func_img, [i, t, k], mask_img, nn)
    return reho_img