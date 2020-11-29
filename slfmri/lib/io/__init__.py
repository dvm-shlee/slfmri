from .afni import AfniIO
from .itksnap import Atlas
from .sitk import nib2sitk, sitk2nib
import json
import pandas as pd
import nibabel as nib
from ..errors import *


def load(file_path: str):
    """
    load available file
    available exts: .nii(.gz), .xls(x), .csv, .tsv, .json

    :param file_path: file want to load
    :type file_path: str
    :return: object
    """
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        img = nib.Nifti1Image.load(file_path)
    else:
        if file_path.endswith('.xls'):
            img = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            img = pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            img = pd.read_table(file_path)
        elif file_path.endswith('.1D'):
            img = pd.read_csv(file_path, header=None, sep=r'\s+')
        elif file_path.endswith('.json'):
            img = json.load(open(file_path))
        else:
            raise Exception('Input filetype is not compatible.')
    return img


def load_volreg(path, mean_radius=9):
    """ Return motion parameter estimated from AFNI's 3dvolreg
    radian values will converted to distance based on given radius.

    :param path:        filepath of 1D data
    :param mean_radius: the distance from aural to central fissure of animal (default: 9mm for rat)
    :return:
    """

    def convert_radian2distance(volreg_, mean_radius_):
        volreg_[['Roll', 'Pitch', 'Yaw']] *= (np.pi / 180 * mean_radius_)
        return volreg_

    volreg = load(path)
    volreg.columns = ['Roll', 'Pitch', 'Yaw', 'dI-S', 'dR-L', 'dA-P']
    r = np.round(np.sqrt(2) * mean_radius)
    return convert_radian2distance(volreg, r)


def save_to_nii(data: np.ndarray,
                niiobj: nib.Nifti1Image,
                fpath: str,
                copy_header=False,
                space='scanner'):
    nii = nib.Nifti2Image(data, niiobj.affine)
    if copy_header:
        nii._header = niiobj.get_header().copy()
    else:
        if space == 'scanner':
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
    nii.to_filename(fpath)


class PathMan:
    def __init__(self, path):
        self._path = path
        self._check_exists()

    def _check_exists(self):
        if not os.path.exists(self._path):
            os.mkdir(self._path)

    def listdir(self, pattern=None):
        result = [p for p in os.listdir(self._path)]
        if pattern is not None:
            result = [p for p in result if re.match(pattern, p)]
        return {i: p for i, p in enumerate(result)}

    def chdir(self, dir_name):
        return PathMan(self(dir_name))

    def __call__(self, fname):
        return os.path.join(self._path, fname)

    def load(self, idx, pattern=None):
        return load(self(self.listdir(pattern=pattern)[idx]))


__all__ = ['AfniIO', 'Atlas', 'load',
           'save_to_nii', 'load_volreg',
           'nib2sitk', 'sitk2nib']
