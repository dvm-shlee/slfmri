import numpy as np


def get_cluster_coordinates(coord, size=1, NN=3, mask=None):
    """
    size: number of voxels want to include from the center
    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    n_voxel = size + 1
    x, y, z = coord
    X = sorted([x + i for i in range(n_voxel)] + [x - i for i in range(n_voxel) if i != 0])
    Y = sorted([y + i for i in range(n_voxel)] + [y - i for i in range(n_voxel) if i != 0])
    Z = sorted([z + i for i in range(n_voxel)] + [z - i for i in range(n_voxel) if i != 0])

    if NN == 1:
        thr = size
    elif NN == 2:
        thr = np.sqrt(np.square([size] * 2).sum())
    elif NN == 3:
        thr = np.sqrt(np.square([size] * 3).sum())
    else:
        raise Exception #TODO: Exception message handler

    all_poss = [(i, j, k) for i in X for j in Y for k in Z]
    output_coord = [c for c in all_poss if cal_distance(coord, c) <= thr]

    if mask is None:
        return output_coord
    else:
        return [c for c in output_coord if c in mask]


def cal_distance(coordA, coordB):
    import numpy as np
    return np.sqrt(np.square(np.diff(np.asarray(list(zip(coordA, coordB))))).sum())


def fwhm2sigma(fwhm):
    import numpy as np
    return fwhm / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma):
    import numpy as np
    return sigma * np.sqrt(8 * np.log(2))