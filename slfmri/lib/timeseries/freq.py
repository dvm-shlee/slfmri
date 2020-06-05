from scipy import sparse
from scipy import signal as signal_
from scipy.sparse import linalg
from typing import Union, List, Optional
from .norm import standardization
from ..errors import *


def bandpass(signal: np.ndarray,
             highpass: Optional[float], lowpass: Optional[float],
             dt: Union[int, float], order: int = 5) -> np.ndarray:
    """ Method to perform bandpass filtering. If only one frequency is given, perform Highpass filter instead.
    Args:
        signal:  time series data
        bandcut: filter frequency range, perform highpass filter if given only one frequency
        dt: sampling time
        order: order of the filter
    Returns:
        filtered signal
    """
    fs = 1.0/float(dt)

    def butter_bandpass(highpass_, lowpass_, fs_, order_):
        nyq = 0.5 * fs_
        if highpass_ and lowpass_:
            highcut = highpass_ / nyq
            lowcut = lowpass_ / nyq
            return signal_.butter(order_, [lowcut, highcut], btype='bandpass', output='ba')
        else:
            if highpass_:
                highcut = highpass_ / nyq
                return signal_.butter(order_, highcut, btype='highpass', output='ba')
            if lowpass_:
                lowcut = lowpass_ / nyq
                return signal_.butter(order_, lowcut, btype='lowpass', output='ba')

    mean = signal.mean()
    std = signal.std()
    normed_signal = standardization(signal)

    ba = butter_bandpass(highpass, lowpass, fs, order_=order)
    y = signal_.lfilter(ba[0], ba[1], normed_signal)
    return np.asarray(y) * std + mean


def als_fit(signal, l: float, p: float, niter: int) -> np.ndarray:
    """ Asymmetric Least Squares Smoothing for Baseline or Envelope fitting
    Args:
        signal: time series data
        l: smoothness, lambda
        p: asymmetry parameter
        niter: number of iteration
    Returns:
        fitted data
    """
    z = None
    l_ = len(signal)
    d_ = sparse.csc_matrix(np.diff(np.eye(l_), 2))
    w = np.ones(l_)
    for i in range(niter):
        w_ = sparse.spdiags(w, 0, l_, l_)
        z_ = w_ + l * d_.dot(d_.transpose())
        z = linalg.spsolve(z_, w * signal)
        w = p * (signal > z) + (1 - p) * (signal < z)
    return np.asarray(z)


def get_phase_angle(signal):
    analytic_signal = signal.hilbert(signal)
    return np.rad2deg(np.angle(analytic_signal))


def periodogram(signal, dt, window='boxcar'):
    fs = 1.0/dt
    f, power = periodogram(signal, fs, window=window)
    return f, power


def welch(signal, dt, window='hann', nperseg=None, noverlap=None):
    fs = 1.0 / dt
    f, power = signal_.welch(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, power


def alff(signal, dt, band=(0.01, 0.1)):
    nperseg = 2 * (1/band[0] / dt)
    if signal.shape[0] < nperseg * 2:
        f, p_spec = signal_.periodogram(signal, dt)
    else:
        f, p_spec = welch(signal, dt, nperseg=nperseg, noverlap=nperseg / 2.0)

    low = np.argmin(abs(f - band[0]))
    high = np.argmin(abs(f - band[1]))

    return p_spec[low:high].mean()
