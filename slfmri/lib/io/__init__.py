from .afni import AfniIO
from .itksnap import Atlas
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


def save_to_nii(func_img: np.ndarray,
                niiobj: nib.Nifti1Image,
                fpath: str):
    nii = nib.Nifti2Image(func_img, niiobj.affine)
    nii._header = niiobj.get_header().copy()
    nii.to_filename(fpath)


__all__ = ['AfniIO',
           'Atlas',
           'load', 'save_to_nii']
