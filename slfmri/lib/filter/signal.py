from ..errors import *
from scipy import signal
from scipy import sparse
from scipy.sparse import linalg as slinalg
from typing import Union, Optional


def bandpass(data: np.ndarray,
             dt: Union[int, float], order: int = 5,
             lowcut: Optional[float] = None,
             highcut: Optional[float] = None,
             output: str = 'sos') -> np.ndarray:
    """ Method to perform bandpass filtering. If only one frequency is given, perform Highpass filter instead.
    Args:
        data:  V x T data where V is voxels and T is time points
        dt: sampling time
        lowcut: filter frequency cut for high-pass filter
        highcut: filter frequency cut for low-pass filter
        order: order of the filter
        output: type of butter filter design
    Returns:
        filtered sig
    """
    fs = 1.0/float(dt)
    nyq = 0.5 * fs
    if lowcut and highcut:
        highcut /= nyq
        lowcut /= nyq
        op = signal.butter(order, [lowcut, highcut], btype='bandpass', output=output)
    else:
        if lowcut:
            lowcut /= nyq
            op = signal.butter(order, lowcut, btype='highpass', output=output)
        elif highcut:
            highcut /= nyq
            op = signal.butter(order, highcut, btype='lowpass', output=output)
        else:
            raise InvalidApproach('Missing filter frequency.')
    if output == 'sos':
        y = signal.sosfilt(op, data)
    elif output == 'ba':
        y = signal.lfilter(op[0], op[1], data)
    elif output == 'zpk':
        raise NotImplemented('Method not implemented yet.')
    else:
        raise InvalidApproach('Invalid metrics.')
    return y


def power_spectral_density(data: np.ndarray,
                           dt: Union[int, float],
                           nperseg: Optional[int] = None,
                           average: str = 'mean'
                           ):
    """ Estimate power spectral density using Welch's metrics

    Args:
        data: V x T data where V is voxels and T is time points
        dt: Sampling time
        nperseg: Length of each segment
        average: Method to use when averaging periodograms

    Returns:
        f : Array of sample frequencies
        pxx : power spectral density of data
    """
    fs = 1.0 / float(dt)
    input_length = data.shape[-1]
    if input_length < 256:
        if nperseg is None:
            nperseg = input_length
    f, pxx = signal.welch(data, fs=fs, window='hann', nperseg=nperseg,
                          scaling='density', average=average)
    return f, pxx


def standardize(data: np.ndarray) -> np.ndarray:
    """ Standardize signal

    Args:
        data: 1d or 2d (V x T)

    Returns:
        standardized data
    """
    dim = data.shape
    if dim == 1:
        sd_data = (data - data.mean()) / data.std()
    else:
        mask_idx = np.nonzero(data.mean(-1))
        masked_data = data[mask_idx]
        masked_sd = np.zeros(masked_data.shape)
        for i, d in enumerate(masked_data):
            masked_sd[i, :] = (d - d.mean()) / d.std()
        sd_data = np.zeros(data.shape)
        sd_data[mask_idx] = masked_sd
    return sd_data


def mode_normalization(data: np.ndarray,
                       mode: Union[int, float] = 100
                       ) -> np.ndarray:
    """ mode normalization

    Args:
        data: 1d or 2d (V x T) data
        mode: mode

    Returns:
        normalized data
    """
    dim = data.shape

    if dim == 1:
        mean = data.mean()
        mn_data = ((data - mean) * mode / mean) + mode
    else:
        mask_idx = np.nonzero(data.mean(-1))
        masked_data = data[mask_idx]
        masked_mn = np.zeros(masked_data.shape)
        mean = masked_data.mean()
        for i, d in enumerate(masked_data):
            masked_mn[i, :] = ((d - mean) * mode / mean) + mode
        mn_data = np.zeros(data.shape)
        mn_data[mask_idx] = masked_mn
    return mn_data


def als_fit(data: np.ndarray,
            l: float, p: float, niter: int) -> np.ndarray:
    """ Asymmetric Least Squares Smoothing for Baseline or Envelope fitting
    TODO: implement for ndarray
    Args:
        data: time series data
        l: smoothness, lambda
        p: asymmetry parameter
        niter: number of iteration
    Returns:
        fitted data
    """
    z = None
    l_ = len(data)
    d_ = sparse.csc_matrix(np.diff(np.eye(l_), 2))
    w = np.ones(l_)
    for i in range(niter):
        w_ = sparse.spdiags(w, 0, l_, l_)
        z_ = w_ + l * d_.dot(d_.transpose())
        z = slinalg.spsolve(z_, w * data)
        w = p * (data > z) + (1 - p) * (data < z)
    return np.asarray(z)


def phase_angle(data):
    analytic_signal = signal.hilbert(data)
    return np.rad2deg(np.angle(analytic_signal))
