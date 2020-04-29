def get_volreg(path, mean_radius=9):
    """ Return motion parameter estimated from AFNI's 3dvolreg
    radian values will converted to distance based on given radius.

    :param path:        filepath of 1D data
    :param mean_radius: the distance from aural to central fissure of animal (default: 9mm for rat)
    :return:
    """
    from .io import load
    import numpy as np

    def convert_radian2distance(volreg, mean_radius):
        volreg[['Roll', 'Pitch', 'Yaw']] *= (np.pi / 180 * mean_radius)
        return volreg

    volreg = load(path)
    volreg.columns = ['Roll', 'Pitch', 'Yaw', 'dI-S', 'dR-L', 'dA-P']
    r = np.round(np.sqrt(2) * mean_radius)
    return convert_radian2distance(volreg, r)


def iszero(signal):
    import numpy as np
    return np.all(signal == 0, axis=0)


def get_funcobj(func, *args, **kwargs):
    """
    Return funcobj to use on 'apply_funcobj' function.
    :param func:    the input function must have 'signal' as first input argument
    :return:
        funcobj
    """
    from collections import namedtuple
    import inspect
    funcobj = namedtuple('FuncObj', ['func', 'args', 'kwargs'])

    # check integrity of function
    if not inspect.isfunction(func):
        raise Exception('Invalid input object.')
    else:
        sig = inspect.signature(func)
        if not 'signal' in sig.parameters:
            raise Exception('Invalid input object.')

    funcobj.func = func
    funcobj.args = args
    funcobj.kwargs = kwargs
    return funcobj


def isfuncobj(funcobj):
    """ Return true if the object is a funcobj object.
    """
    dtype = type(funcobj)
    bases = dtype.__bases__
    if len(bases) != 1 or not isinstance(bases[0], object): return False
    fields = getattr(funcobj, '_fields', None)
    if not isinstance(fields, tuple): return False
    return all([f in ['func', 'args', 'kwargs'] for f in fields])


def apply_funcobj(funcobj, func_img, mask_img=None):
    """
    This function apply given funcobj(s) to input data on time domain.
    If the list of funcobjs are given, funcobjs will be applied serial manner

    :param funcobj:         a funcobj or list of funcobjs
    :param func_img:        3D+time data
    :param mask_img:        binary 3D data for masking (default: None)
    :return processed_img:  processed image data with given function
    """
    import numpy as np
    # Check data integrity
    dim = len(func_img.shape)
    if dim < 4 or dim > 4:
        raise Exception('This function only process 3D+time data.')

    # encaptulate with list for iteration
    if not isinstance(funcobj, list):
        funcobj = [funcobj]

    # Apply function to time domain
    if mask_img is not None:
        indices = np.transpose(np.nonzero(mask_img))
    else:
        indices = np.transpose(np.nonzero(func_img.mean(-1)))

    processed_img = np.zeros(func_img.shape[:3])
    for n, (i, j, k) in enumerate(indices):
        td_data = func_img[i, j, k, :]
        for f in funcobj:
            if not isfuncobj(f):
                # Check integrity of given function
                raise Exception('The input is not funcobj.')
            else:
                td_data = f.func(td_data, *f.args, **f.kwargs)

        if n == 0:
            if isinstance(td_data, np.ndarray):
                length = td_data.shape[0]
                processed_img = processed_img[:,:,:,np.newaxis]
                processed_img = np.concatenate([processed_img] * length, axis=-1)
        processed_img[i, j, k] = td_data
    return processed_img


def mkdir(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)


def get_filepath(path, ext=None):
    import os
    if ext:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and ext in f])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def get_subdirpath(path):
    import os
    return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])


def joinpath(*args):
    import os
    return os.path.join(*args)

def get_cluster_coordinates(coord, size=1, NN=3, mask=None):
    """
    size: number of voxels want to include from the center
    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    import numpy as np
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

    if mask == None:
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