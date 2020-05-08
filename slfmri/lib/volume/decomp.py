from ..errors import *
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from ..timeseries import standardization


def estimate_1st_pc(func_img, mask_img, n_comp=5):
    pca = PCA(n_components=n_comp)
    ts_signals = func_img[np.nonzero(mask_img)]
    pca.fit(ts_signals)
    principle_comp = standardization(pca.components_[0])
    r_, p_ = pearsonr(standardization(ts_signals.mean(0)), principle_comp)
    if r_ < 0:
        # invert sign if PC is negative correlated with the original signal.
        principle_comp = principle_comp * -1
    return principle_comp