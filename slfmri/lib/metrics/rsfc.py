from ..errors import *
from ..filter import power_spectral_density
from ..utils import get_cluster_coordinates
from scipy import stats
from typing import Union, Optional
import pandas as pd


def dvars(data):
    mask_idx = np.nonzero(data.mean(-1))
    diff_img = np.diff(data[mask_idx], axis=-1)
    dvars_ = np.sqrt(np.square(diff_img).mean(0))
    return np.insert(dvars_, 0, 0)


def bold_mean_std(data):
    mask_idx = np.nonzero(data.mean(-1))
    masked_data = data[mask_idx]
    for d in masked_data: # demean
        d -= d.mean()
    return masked_data.mean(0), masked_data.std(0)


def framewise_displacements(volreg):
    """
    Calculate volume displacement from motion parameter
    """
    output = dict()
    columns = volreg.columns
    # Framewise displacement
    output['FD'] = np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute rotational displacement
    output['ARD'] = np.abs(np.insert(np.diff(volreg[columns[:3]], axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute translational displacement
    output['ATD'] = np.abs(np.insert(np.diff(volreg[columns[3:]], axis=0), 0, 0, axis=0)).sum(axis=1)
    return pd.DataFrame(output)


def amplitude_low_freq_fluctuation(data: np.ndarray,
         dt: Union[int, float],
         lowcut: Union[float],
         highcut: Union[float]):
    """ Amplitude of Low Frequency Fluctuation

    Args:
        data: V x T
        dt: sampling time
        lowcut: cut frequency for highpass filter
        highcut: cut frequency for lowpass filter

    Returns:
        ALFF
    """
    f, pxx = power_spectral_density(data, dt=dt)
    alff = pxx[(f >= lowcut) & (f <= highcut)].sum(-1)
    return alff


def regional_homogeneity(data, nn=3):
    def kandallw(data, coord, nn_level):
        """
        This function calculate regional homogeneity of the coordinate
        using neighboring voxels in given cluster size.

        The tied rank adjustment are performed using scipy's rankdata metrics

        nn_level: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
        """
        mask = np.transpose(np.nonzero(data.mean(-1)))
        n_ = data.shape[-1]  # number of "objects"
        indices = get_cluster_coordinates(coord, size=1, nn_level=nn_level, mask=mask)
        m_ = len(indices)  # number of "judge"

        # Perform rank judgements
        rank_matrix = np.zeros([n_, m_])
        for idx, neighbor in enumerate(indices):
            i, j, k = neighbor
            try:
                rank_matrix[:, idx] = stats.rankdata(data[i, j, k, :])
            except IndexError:
                # This exception handle the case that coordinate of neighbor is located outside of the matrix
                pass
            except:
                raise UnexpectedError
        ranks = rank_matrix.sum(1)

        # Calculate the mean value of these total ranks
        mean_ranks = ranks.mean()

        # Calculate sum of squared deviations (SSD)
        ssd_ = np.square(ranks - mean_ranks).sum()

        # Calculate Kendall's W
        w_ = 12 * ssd_ / (m_ ** 2 * (n_ ** 3 - n_))
        return w_

    mask_idx = np.transpose(np.nonzero(data.mean(-1)))
    reho_img = np.zeros(data.shape[:3])
    for i, t, k in mask_idx:
        reho_img[i, t, k] = kandallw(data, [i, t, k], nn)
    return reho_img


def tsnr(data: np.ndarray) -> Union[np.ndarray, int]:
    """ calculate temporal snr

    Args:
        data: 1d or 2d (V x T) data

    Returns:
        tsnr
    """
    dim = data.shape
    if dim == 1:
        tsnr = data.mean() / data.std()
    else:
        mask_idx = np.nonzero(data.mean(-1))
        masked_data = data[mask_idx]
        masked_tsnr = np.zeros(masked_data.shape)
        for i, d in enumerate(masked_data):
            masked_tsnr[i, :] = d.mean() / d.std()
        tsnr = np.zeros(data.shape)
        tsnr[mask_idx] = masked_tsnr
    return tsnr