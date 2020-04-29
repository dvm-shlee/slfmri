def kandallw(func_img, coord, mask_img=None, NN=3):
    """
    This function calculate regional homogeneity of the coordinate
    using neighboring voxels in given cluster size.

    The tied rank adjustment are performed using scipy's rankdata method

    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    from scipy.stats import rankdata
    from .tools import get_cluster_coordinates
    import numpy as np

    if mask_img is not None:
        mask = np.transpose(np.nonzero(mask_img))
    else:
        mask = None

    N = func_img.shape[-1]      # number of "objects"
    indices = get_cluster_coordinates(coord, size=1, NN=NN, mask=mask)
    M = len(indices)            # number of "judge"

    # Perform rank judgements
    rank_matrix = np.zeros([N, M])
    for idx, neighbor in enumerate(indices):
        i, j, k = neighbor
        try:
            rank_matrix[:, idx] = rankdata(func_img[i, j, k, :])
        except:
            # This exception handle the case that coordinate of neighbor is located outside of the matrix
            pass
    ranks = rank_matrix.sum(1)

    # Calculate the mean value of these total ranks
    mean_ranks = ranks.mean()

    # Calculate sum of squiared deviations (SSD)
    SSD = np.square(ranks - mean_ranks).sum()

    # Calculate Kendall's W
    W = 12 * SSD / (M ** 2 * (N ** 3 - N))
    return W


def reho(func_img, mask_img=None, nn=3):
    import numpy as np
    if mask_img is not None:
        indices = np.transpose(np.nonzero(mask_img))
    else:
        indices = np.transpose(np.nonzero(func_img.mean(-1)))
    reho_img = np.zeros(func_img.shape[:3])
    for i, t, k in indices:
        reho_img[i, t, k] = kandallw(func_img, [i, t, k], mask_img, nn)
    return reho_img


def periodogram(signal, dt, window='boxcar'):
    from scipy.signal import periodogram

    fs = 1.0/dt
    f, power = periodogram(signal, fs, window=window)
    return f, power


def welch(signal, dt, window='hann', nperseg=None, noverlap=None):
    from scipy.signal import welch
    fs = 1.0 / dt
    f, power = welch(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, power


def alff(signal, dt, band=(0.01, 0.1)):
    import numpy as np
    # from scipy.integrate import simps

    nperseg = 2 * (1/band[0] / dt)
    if signal.shape[0] < nperseg * 2:
        f, Pspec = periodogram(signal, dt)
    else:
        f, Pspec = welch(signal, dt, nperseg=nperseg, noverlap=nperseg / 2.0)

    low = np.argmin(abs(f - band[0]))
    high = np.argmin(abs(f - band[1]))

    # return simps(Pspec[low:high], dx=1)
    return Pspec[low:high].mean()


def get_phase_angle(signal):
    import numpy as np
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal)
    return np.rad2deg(np.angle(analytic_signal))