from ..errors import *
from ..filter import power_spectral_density
from ..utils import get_cluster_coordinates
from typing import Union, Optional
import pandas as pd


def dvars(data, io_handler=sys.stdout):
    io_handler.write('Calculating DVARS...')
    mask_idx = np.nonzero(data.mean(-1))
    diff_img = np.diff(data[mask_idx], axis=-1)
    dvars_ = np.sqrt(np.square(diff_img).mean(0))
    io_handler.write('[Done]\n')
    return np.insert(dvars_, 0, 0)


def bold_mean_std(data, io_handler=sys.stdout):
    io_handler.write('Calculating Mean and STD...')
    mask_idx = np.nonzero(data.mean(-1))
    masked_data = data[mask_idx]
    masked_data = (masked_data.T - masked_data.mean(1).T).T
    io_handler.write('[Done]\n')
    return masked_data.mean(0), masked_data.std(0)


def framewise_displacements(volreg, io_handler=sys.stdout):
    """
    Calculate volume displacement from motion parameter
    """
    output = dict()
    columns = volreg.columns
    # Framewise displacement
    io_handler.write('Calculating Displacement from motion parameters...')
    output['FD'] = np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute rotational displacement
    output['ARD'] = np.abs(np.insert(np.diff(volreg[columns[:3]], axis=0), 0, 0, axis=0)).sum(axis=1)
    # Absolute translational displacement
    output['ATD'] = np.abs(np.insert(np.diff(volreg[columns[3:]], axis=0), 0, 0, axis=0)).sum(axis=1)
    io_handler.write('[Done]\n')
    return pd.DataFrame(output)


def amplitude_low_freq_fluctuation(data: np.ndarray,
                                   dt: Union[int, float],
                                   lowcut: Union[float], highcut: Union[float],
                                   pval: Optional[float]=None,
                                   fraction: bool=False,
                                   io_handler=sys.stdout):
    """ Amplitude of Low Frequency Fluctuation

    Args:
        data: V x T
        dt: sampling time
        lowcut: cut frequency for highpass filter
        highcut: cut frequency for lowpass filter

    Returns:
        ALFF
    """
    io_handler.write('Calculating ALFF...')
    f, pxx = power_spectral_density(data, dt=dt)
    alff = pxx[..., (f >= lowcut) & (f <= highcut)].sum(-1)
    if fraction:
        alff[np.nonzero(pxx.sum(-1))] /= pxx.sum(-1)[np.nonzero(pxx.sum(-1))]
    io_handler.write('[Done]\n')
    return alff


def kendal_w(data, nn=None):
    from scipy.stats import rankdata
    m, n = data.shape
    if nn is not None:
        if nn == 1:
            n = 7
        elif nn == 2:
            n = 19
        elif nn == 3:
            n = 27
        else:
            raise UnexpectedError('out of range')
    if m != 0:
        ranks = np.apply_along_axis(rankdata, 1, data).sum(0)
        mean_ranks = ranks.mean()
        ssd = np.square(ranks - mean_ranks).sum()
        w = 12 * ssd / (m ** 2 * (n ** 3 - n))
        return w
    else:
        return 0


def regional_homogeneity(data, nn=3, io_handler=sys.stdout):
    from functools import partial
    mask_idx = np.nonzero(data.mean(-1))
    gcc = partial(get_cluster_coordinates, size=1, nn_level=nn)
    io_handler.write('Extracting nearest coordinates set...')
    all_coords_set = np.apply_along_axis(gcc, 0, np.array(mask_idx)).T
    io_handler.write('[Done]\n')
    masked_reho = np.zeros(all_coords_set.shape[0])

    n_voxels = all_coords_set.shape[0]
    progress = 1
    io_handler.write('Calculating regional homogeneity...\n')
    for i, coord in enumerate(all_coords_set):
        # filter outbound coordinate
        c_msk = []
        for j, arr in enumerate(coord):
            s = data.shape[j]
            c_msk.append(np.nonzero(arr > s - 1)[0])
        coord_flt = [f for f in range(coord.shape[-1]) if f not in list(set(np.concatenate(c_msk, 0)))]
        coord = coord[:, coord_flt]
        cd = data[tuple(coord)]
        masked_cd = cd[np.nonzero(cd.mean(-1))]
        masked_reho[i] = kendal_w(masked_cd, nn=nn)
        if (i / n_voxels) * 10 >= progress:
            io_handler.write(f'{progress}..')
            progress += 1
        if i == (n_voxels - 1):
            io_handler.write('10 [Done]\n')

    reho = np.zeros(data.shape[:3])
    reho[mask_idx] = masked_reho
    return reho


def tsnr(data: np.ndarray, io_handler=sys.stdout) -> Union[np.ndarray, int]:
    """ calculate temporal snr

    Args:
        data: 1d or 2d (V x T) data

    Returns:
        tsnr
    """
    io_handler.write('Calculating tSNR...')
    dim = data.shape
    if dim == 1:
        tsnr = data.mean() / data.std()
    else:
        mean = data.mean(-1)
        std = data.std(-1)
        tsnr = np.zeros(mean.shape)
        masked_mean = mean[np.nonzero(std)]
        masked_std = std[np.nonzero(std)]
        tsnr[np.nonzero(std)] = masked_mean / masked_std
    io_handler.write('[Done]\n')
    return tsnr


def connectivity_strength(x: np.ndarray, y: Optional[np.ndarray] = None,
                          pval: Optional[float] = None,
                          pos=False, abs=False) -> np.ndarray:
    from ..stats.corr import corr, corr_with, r_to_t
    if y is None:
        r = corr(x)
        r[np.nonzero(np.eye(r.shape[0]))] = 0
    else:
        r = corr_with(x, y)

    if pos:
        r[r < 0] = 0
    if abs:
        r = np.abs(r)
    if pval is not None:
        t, p = r_to_t(r, x.shape[-1])
        r[p >= pval] = 0
    r[np.nonzero(np.eye(r.shape[0]))] = 0
    return r.sum(-1)