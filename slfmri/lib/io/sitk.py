from ..errors import *
from .orient import to_matvec, from_matvec
import nibabel as nib
import SimpleITK as sitk
from typing import Optional


def affine_nii2sitk(affine, resol):
    affine = np.matmul(np.diag([-1, -1, 1, 1]), affine)
    direction, origin = to_matvec(affine)
    direction = direction.dot(np.linalg.inv(np.diag(resol)))
    return np.round(direction, decimals=3), origin


def affine_sitk2nii(direction, origin, resol):
    direction = direction.dot(np.diag(resol))
    affine = from_matvec(direction, origin)
    affine = np.matmul(np.diag([-1, -1, 1, 1]), affine)
    return np.round(affine, decimals=3)


def get_3dvol(sitk_img, frame=0):
    # 4d to 3d
    size = list(sitk_img.GetSize())
    if len(size) == 4:
        size[3] = 0
        index = [0, 0, 0, frame]
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(sitk_img)
    else:
        raise InvalidApproach('Data is not 4D.')
    return image


def nib2sitk(nii_img: nib.Nifti1Image, is_vector: Optional[bool] = False) -> (sitk.Image, nib.Nifti1Header):
    img = sitk.GetImageFromArray(np.asarray(nii_img.dataobj).T, isVector=is_vector)
    header = nii_img.header.copy()
    resol = nii_img.header['pixdim'][1:5]
    direction_, origin_ = affine_nii2sitk(nii_img.affine.copy(), resol[:3])
    if len(img.GetSize()) > 3:
        direction = np.eye(4)
        direction[:3, :3] = direction_
        direction = direction.flatten()
        origin = np.zeros([4])
        origin[:3] = origin_
    else:
        direction = direction_.flatten()
        origin = origin_
    img.SetDirection(direction)
    img.SetOrigin(origin)
    img.SetSpacing(resol.tolist())
    return img, header


def sitk2nib(sitk_img: sitk.Image, header: Optional[nib.Nifti1Header] = None) -> nib.Nifti1Image:
    dataobj = sitk.GetArrayFromImage(sitk_img).T
    resol = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = np.asarray(sitk_img.GetDirection())
    if len(resol) > 3:
        direction = direction.reshape([4, 4])[:3, :3]
    else:
        direction = direction.reshape([3, 3])
    affine = affine_sitk2nii(direction, origin[:3], resol[:3])
    img = nib.Nifti1Image(dataobj, affine)
    if header is None and len(resol) > 3:
        img.header['pixdim'][4] = resol[3]
        img.header.set_xyzt_units(xyz=2, t=8)
    else:
        img._header = header.copy()
    return img
