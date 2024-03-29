from .errors import *    # os module imported from here


def random_rgb():
    levels = range(32,256,32)
    return tuple(np.random.choice(levels) for _ in range(3))


def get_rgb_tuple(r: int, g: int, b: int) -> (float, float, float):
    return r / 255., g / 255., b / 255.


def iszero(data):
    return np.all(data == 0, axis=0)


def get_cluster_coordinates(coord, size=1, nn_level=3):
    """
    size: number of voxels want to include from the center
    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    n_voxel = size + 1
    x, y, z = coord
    x_ = sorted([x + i for i in range(n_voxel)] + [x - i for i in range(n_voxel) if i != 0])
    y_ = sorted([y + i for i in range(n_voxel)] + [y - i for i in range(n_voxel) if i != 0])
    z_ = sorted([z + i for i in range(n_voxel)] + [z - i for i in range(n_voxel) if i != 0])

    if nn_level == 1:
        thr = size
    elif nn_level == 2:
        thr = np.sqrt(np.square([size] * 2).sum())
    elif nn_level == 3:
        thr = np.sqrt(np.square([size] * 3).sum())
    else:
        raise ValueError('[nn_level] only accept a value in [1, 2, 3]')

    all_poss = [(i, j, k) for i in x_ for j in y_ for k in z_]
    output_coord = [c for c in all_poss if cal_distance(coord, c) <= thr]

    return output_coord


def cal_distance(coord_a, coord_b):
    return np.sqrt(np.square(np.diff(np.asarray(list(zip(coord_a, coord_b))))).sum())

