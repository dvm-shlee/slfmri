from ..utils import iszero


def linear_regression(signal, estimator, design_matrix):
    if iszero(signal):
        return signal
    else:
        # Applying regression denoising
        model = estimator()
        model.fit(design_matrix, signal)
        return model

    
def bandpass(signal, bandcut, dt, order=5):
    """
    Temporal frequency filter

    :param signal:      time series data
    :param bandcut:     filter frequency range, perform highpass filter if given only one frequency
    :param dt:          sampling time
    :param order:       order of the filter
    :return:            filtered data
    """
    import numpy as np
    from scipy.signal import butter, lfilter
    from .norm import standardization

    fs = 1.0/dt

    def butter_bandpass(cut_freqs, fs, order):
        nyq = 0.5 * fs
        if isinstance(cut_freqs, list) and len(cut_freqs) == 2:
            lowcut = cut_freqs[0] / nyq
            highcut = cut_freqs[1] / nyq
            return butter(order, [lowcut, highcut], btype='bandpass', output='ba')
        else:
            try:
                onesidecut = cut_freqs / nyq
                return butter(order, onesidecut, btype='highpass', output='ba')
            except:
                raise Exception('Wrong cut frequency value(s).')

    mean = signal.mean()
    std = signal.std()
    normed_signal = standardization(signal)

    ba = butter_bandpass(bandcut, fs, order=order)
    y = lfilter(ba[0], ba[1], normed_signal)
    return np.asarray(y) * std + mean


def estimate_pc(func_img, mask_img, n_comp=5):
    import numpy as np
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    from .norm import standardization

    pca = PCA(n_components=n_comp)
    ts_signals = func_img[np.nonzero(mask_img)]
    pca.fit(ts_signals)
    principle_comp = standardization(pca.components_[0])
    r_, p_ = pearsonr(standardization(ts_signals.mean(0)), principle_comp)
    if r_ < 0:
        # invert sign if PC is negative correlated with the original signal.
        principle_comp = principle_comp * -1
    return principle_comp


def polynomial_fit(signal, estimator, order=3):
    """
    Estimate polynomial curve fit for data

    :param signal: time series data
    :param estimator: estimator for linear regression
    :param order: order of polynomial curve
    :return: fitted curve
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    model = make_pipeline(PolynomialFeatures(order), estimator())
    x = np.linspace(0, (len(signal) - 1) * 2, len(signal))
    X = x[:, np.newaxis]
    model.fit(X, signal)
    return np.asarray(model.predict(X))


def nuisance_regression(signal, estimator, ort=None, order=3):
    import pandas as pd
    import numpy as np
    from .norm import standardization

    polort = pd.DataFrame(polynomial_fit(signal, estimator, order=order))

    if ort is None:
        design_matrix = polort
    else:
        if isinstance(ort, list):
            ort_list = [polort]
            for o in ort:
                o = pd.DataFrame(o)
                ort_list.extend([o, o.diff().fillna(0)])
            design_matrix = pd.concat(ort_list, axis=1, ignore_index=True)
        else:
            ort = pd.DataFrame(ort)
            design_matrix = pd.concat([polort, ort, ort.diff().fillna(0)],
                                      axis=1, ignore_index=True)
    design_matrix = standardization(design_matrix, axis=0)

    model = linear_regression(signal, estimator, design_matrix)
    if isinstance(model, np.ndarray):
        return model
    else:
        regressor = model.predict(design_matrix)
        regressor -= regressor.mean()
        return np.asarray(signal - regressor)


def als_fit(data, l, p, niter):
    """
    Asymmetric Least Squares Smoothing for Baseline or Envelope fitting

    :param data: time series data
    :param l: smoothness, lambda
    :param p: assymetry parameter
    :param niter: number of iteration
    :return: fitted data
    """
    import numpy as np
    from scipy.sparse import linalg
    from scipy import sparse
    z = None
    L = len(data)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + l * D.dot(D.transpose())
        z = linalg.spsolve(Z, w * data)
        w = p * (data > z) + (1 - p) * (data < z)
    return np.asarray(z)
