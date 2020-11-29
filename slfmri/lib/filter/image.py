""" collection of function to manipulate medical image """
from ..errors import *
import SimpleITK as sitk
from ..io.sitk import get_3dvol
from typing import Optional, IO


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def gaussian_smoothing(sitk_img: sitk.Image, fwhm: float,
                       io_handler: Optional[IO] = None) -> sitk.Image:
    if io_handler is None:
        import sys
        io_handler = sys.stdout
    dim = sitk_img.GetDimension()
    sigma = fwhm2sigma(fwhm)
    parser = []
    if dim > 3:
        num_frames = sitk_img.GetSize()[-1]
        progress = 1
        for f in range(num_frames):
            vol_img = get_3dvol(sitk_img, f)
            parser.append(sitk.SmoothingRecursiveGaussian(vol_img, sigma))
            if (f/num_frames) * 10 >= progress:
                io_handler.write(f'{progress}..')
                progress += 1
            if f == (num_frames - 1):
                io_handler.write('10 [Done]\n')
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