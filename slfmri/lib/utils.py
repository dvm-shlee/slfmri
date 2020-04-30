from .io import load
import numpy as np
from collections import namedtuple
import inspect
import os
from .errors import *


def get_rgb_tuple(r: int, g: int, b: int) -> (float, float, float):
    return r / 255., g / 255., b / 255.


def get_volreg(path, mean_radius=9):
    """ Return motion parameter estimated from AFNI's 3dvolreg
    radian values will converted to distance based on given radius.

    :param path:        filepath of 1D data
    :param mean_radius: the distance from aural to central fissure of animal (default: 9mm for rat)
    :return:
    """

    def convert_radian2distance(volreg, mean_radius):
        volreg[['Roll', 'Pitch', 'Yaw']] *= (np.pi / 180 * mean_radius)
        return volreg
    volreg = load(path)
    volreg.columns = ['Roll', 'Pitch', 'Yaw', 'dI-S', 'dR-L', 'dA-P']
    r = np.round(np.sqrt(2) * mean_radius)
    return convert_radian2distance(volreg, r)


def iszero(signal):
    return np.all(signal == 0, axis=0)


def get_funcobj(func, *args, **kwargs):
    """
    Return funcobj to use on 'apply_funcobj' function.
    :param func:    the input function must have 'signal' as first input argument
    :return:
        funcobj
    """
    funcobj = namedtuple('FuncObj', ['func', 'args', 'kwargs'])

    # check integrity of function
    if not inspect.isfunction(func):
        raise InvalidApproach('Invalid input object.')
    else:
        sig = inspect.signature(func)
        if not 'signal' in sig.parameters:
            raise InvalidApproach('Invalid input object.')

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
    :return processed_img:  processed space data with given function
    """
    # Check data integrity
    dim = len(func_img.shape)
    if dim < 4 or dim > 4:
        raise InvalidApproach('This function only process 3D+time data.')

    # decorate with list for iteration
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
                processed_img = processed_img[..., np.newaxis]
                processed_img = np.concatenate([processed_img] * length, axis=-1)
        processed_img[i, j, k] = td_data
    return processed_img


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_filepath(path, ext=None):
    if ext:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and ext in f])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


def get_subdirpath(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])


def joinpath(*args):
    return os.path.join(*args)


