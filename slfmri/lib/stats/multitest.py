from ..errors import *


def bonfferoni_correction(r, size, pval=0.05):
    """
    r: 1d matrix with the shape of N where N is number of comparisons
    """
    from .corr import r_to_t
    t, p = r_to_t(r, size)
    return p < (pval / r.shape[0])


