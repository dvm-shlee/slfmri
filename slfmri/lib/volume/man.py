""" collection of function to manipulate medical image """
from ..errors import *
from .orient import to_matvec, from_matvec
from .tools import fwhm2sigma
import nibabel as nib
import SimpleITK as sitk
from typing import Optional, IO


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
        img.header.set_xyzt_units('mm', 'sec')
    else:
        img._header = header.copy()
    return img


def gaussian_smoothing(sitk_img: sitk.Image, fwhm: float,
                       io_handler: Optional[IO] = None) -> sitk.Image:
    dim = sitk_img.GetDimension()
    sigma = fwhm2sigma(float(fwhm))
    parser = []
    if dim > 3:
        num_frames = sitk_img.GetSize()[-1]
        progress = 1
        for f in range(num_frames):
            vol_img = get_3dvol(sitk_img, f)
            parser.append(sitk.SmoothingRecursiveGaussian(vol_img, sigma))
            if io_handler is not None:
                if (f/num_frames) * 10 >= progress:
                    io_handler.write(f'{progress}..')
                    progress += 1
                if f == (num_frames - 1):
                    io_handler.write('10 [Done]\n')
            else:
                if (f/num_frames) * 10 >= progress:
                    print(progress, end='..', file=io_handler)
                    progress += 1
                if f == (num_frames - 1):
                    print('10 [Done]', file=io_handler)
        smoothed_img = sitk.JoinSeries(parser)
        smoothed_img.CopyInformation(sitk_img)
        return smoothed_img
    return sitk.SmoothingRecursiveGaussian(sitk_img, sigma)


def n4_biasfield_correction(sitk_img: sitk.Image,
                            mask_img: Optional[sitk.Image] = None,
                            num_iter: Optional[int] = None,
                            num_fit_level: Optional[int] = None) -> sitk.Image:

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if num_iter is not None:
        if num_fit_level is not None:
            corrector.SetMaximumNumberOfIterations(num_iter * num_fit_level)
        else:
            corrector.SetMaximumNumberOfIterations(num_iter)
    input_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    if mask_img is not None:
        return corrector.Execute(input_img, mask_img)
    else:
        return corrector.Execute(input_img)


if __name__ == '__main__':
    import sys
    file_path = 'examples/camri_isotropic_epi.nii.gz'
    output_path = 'examples/test_outputs'
    sitk_img_ = sitk.ReadImage(file_path)

    # vol_img_ = get_3dvol(sitk_img_, 0)
    # n4_img_ = n4_biasfield_correction(vol_img_)
    # s_img_ = gaussian_smoothing(sitk_img_, fwhm=0.5, io_handler=sys.stderr)
    # sitk.WriteImage(s_img_, os.path.join(output_path, 'smoothed_img.nii.gz'))
