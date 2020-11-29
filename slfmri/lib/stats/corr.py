from ..errors import *
from scipy.signal import hilbert


def pearsonr(x, y):
    covariance = ((x - x.mean()) * (y - y.mean())).sum()
    denominator = np.sqrt(np.square(x-x.mean()).sum()) * np.sqrt(np.square(y-y.mean()).sum())
    return covariance / denominator


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