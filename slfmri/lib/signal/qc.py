from ..errors import *
from typing import Union


def tsnr(signal: np.ndarray) -> Union[np.ndarray, int]:
    if signal.mean() == 0 or signal.std() == 0:
        return 0
    else:
        return signal.mean() / signal.std()


def framewise_displacements(volreg):
    """
    This function calculate volume displacement from motion parameter
    """
    import numpy as np
    import pandas as pd

    output = dict()
    columns = volreg.columns
    # Framewise displacement
    output['FD'] = np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute rotational displacement
    output['ARD'] = np.abs(np.insert(np.diff(volreg[columns[:3]], axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute translational displacement
    output['ATD'] = np.abs(np.insert(np.diff(volreg[columns[3:]], axis=0), 0, 0, axis=0)).sum(axis=1)
    return pd.DataFrame(output)


def convert_radian2distance(volreg, mean_radius):
    """
    :param volreg:
    :param mean_radius:
    :return:
    """
    import numpy as np
    volreg[['Roll', 'Pitch', 'Yaw']] *= (np.pi / 180 * mean_radius)
    return volreg
