from scipy import stats
from ..volume.tools import get_cluster_coordinates
from ..errors import *


def kandallw(func_img, coord, mask_img=None, nn_level=3):
    """
    This function calculate regional homogeneity of the coordinate
    using neighboring voxels in given cluster size.

    The tied rank adjustment are performed using scipy's rankdata method

    nn_level: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    if mask_img is not None:
        mask = np.transpose(np.nonzero(mask_img))
    else:
        mask = None

    n_ = func_img.shape[-1]      # number of "objects"
    indices = get_cluster_coordinates(coord, size=1, nn_level=nn_level, mask=mask)
    m_ = len(indices)            # number of "judge"

    # Perform rank judgements
    rank_matrix = np.zeros([n_, m_])
    for idx, neighbor in enumerate(indices):
        i, j, k = neighbor
        try:
            rank_matrix[:, idx] = stats.rankdata(func_img[i, j, k, :])
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
