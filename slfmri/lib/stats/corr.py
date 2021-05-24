from ..errors import *
from scipy.signal import hilbert


def corr(x):
    vals = np.zeros(x.shape)
    mask = np.nonzero(x.std(-1))

    try:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T
    except:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T.astype(np.float32)
    r = np.dot(vals, vals.T) / vals.shape[-1]
    return r


def corr_with(x, y):

    val_x = np.zeros(x.shape)
    val_y = np.zeros(y.shape)

    x_mask = np.nonzero(x.std(-1))
    y_mask = np.nonzero(y.std(-1))
    try:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T
    except:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T.astype(np.float32)
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T.astype(np.float32)
    r = np.dot(val_x, val_y.T) / x.shape[-1]
    return r


def phase_locking_value(x, y):
    x_phase = np.angle(hilbert(x), deg=False)
    y_phase = np.angle(hilbert(y), deg=False)
    angle_diff = x_phase - y_phase
    return abs(np.exp(1j*angle_diff).mean()), angle_diff


def const_maxcorr(df, dt, max_lag):
    """
    df: DataFrame (TxR) where T is number of timepoints, R is number of ROIs
    dt: sampling interval (unit: sec)
    max_lag: constrained maximum lag (unit: sec)

    Returns:
        max_corr:
    """
    import pandas as pd
    max_lag = int(max_lag / dt)
    all_lags = np.arange(-max_lag, max_lag + 1)
    max_corr = pd.DataFrame(np.zeros([df.shape[-1]] * 2), index=df.columns,
                            columns=df.columns)
    max_corr_lag = max_corr.copy()

    for col_id1, ts1 in df.iteritems():
        cross_corr = np.zeros([len(all_lags), len(df.columns)])
        for col_id2, ts2 in df.iteritems():
            for lag_id, lag in enumerate(all_lags):
                cross_corr[lag_id, col_id2] = ts1.corr(ts2.shift(lag))
        max_lag_idxs = abs(cross_corr).argmax(0)
        for col_id2, max_lag_idx in enumerate(max_lag_idxs):
            max_corr.loc[col_id1, col_id2] = cross_corr[max_lag_idx, col_id2]
            max_corr_lag.loc[col_id1, col_id2] = all_lags[max_lag_idx]
    return max_corr, max_corr_lag


def r_to_t(r, size):
    """
    calculate T and correspond p-value from given correlation matrix
    """
    from scipy.stats import t
    try:
        tval = r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))
        pval = 1 - t.cdf(tval, size - 2)
    except:
        r = r.astype(np.float32)
        tval = (r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))).astype(np.float32)
        pval = (1 - t.cdf(tval, size - 2)).astype(np.float32)
    return tval, pval