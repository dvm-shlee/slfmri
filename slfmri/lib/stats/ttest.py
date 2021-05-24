from ..errors import *


def ttest(data, model):
    from slfmri import filter as slflt
    from scipy import stats
    predicted, coef = slflt.linear_regression(data, model, method='svd', return_beta=True)

    dof = model.shape[0] - model.shape[1]
    mse = np.square(predicted.sum(-1) - data).sum(-1) / float(dof)
    se = np.sqrt((mse * np.concatenate([np.linalg.inv(np.dot(model.T, model)).diagonal()[:, np.newaxis]], axis=-1)).T)

    t = coef.copy()
    if model.shape[-1] == 1:
        # one sample t-test
        t -= t.mean()  # against population mean
    t[se == 0] = 0
    t[np.nonzero(se)] /= se[np.nonzero(se)]
    p = 2 * (1 - stats.t.cdf(abs(t), df=dof))
    return coef, t, p


def onesample_ttest_perm(data, pval=0.05, nperm=5000, twosided=False):
    from tqdm.notebook import tqdm
    model = np.ones([data.shape[1], 1])
    b, t_o, p_o = ttest(data, model)

    tmax = np.zeros(nperm)
    tmin = np.zeros(nperm)
    for p in tqdm(range(nperm)):
        model_perm = np.c_[model, np.random.choice([1, -1], data.shape[-1], replace=True)]
        b_p, t_p, p_p = ttest(data, model_perm)
        tmax[p] = t_p[:, 1].max()
        tmin[p] = t_p[:, 1].min()
    if twosided:
        pp_perm = np.zeros(data.shape[0])
        np_perm = pp_perm.copy()
        for i in range(data.shape[0]):
            pp_perm[i] = (tmax >= t_o[i, 0]).astype(int).sum() / nperm
            np_perm[i] = (tmin <= t_o[i, 0]).astype(int).sum() / nperm
        t_o[(pp_perm >= pval / 2) & (np_perm >= pval / 2)] = 0
    else:
        p_perm = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            p_perm[i] = (tmax >= t_o[i, 0]).astype(int).sum() / nperm
        t_o[p_perm >= pval] = 0
    return b