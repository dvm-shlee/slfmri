from ..errors import *


def tsnr(signal: np.ndarray) -> np.ndarray:
    return signal.mean() / signal.std()


def mparam_fd(volreg):
    """ Frame-wise displacement """
    return np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)


def mparam_ard(volreg):
    """ Absolute rotational displacement """
    return np.abs(np.insert(np.diff(volreg[volreg.columns[:3]], axis=0),
                            0, 0, axis=0)).sum(axis=1)


def mparam_atd(volreg):
    return np.abs(np.insert(np.diff(volreg[volreg.columns[3:]], axis=0),
                            0, 0, axis=0)).sum(axis=1)
