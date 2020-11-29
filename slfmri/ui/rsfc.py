from ..lib.errors import *
from ..lib.io.orient import decomp_dataobj, save_to_nib, nib
from typing import Optional, Union, IO


def dvars(nib_img: nib.Nifti1Image,
          mask: Optional[nib.Nifti1Image] = None):
    from ..lib.metrics import dvars as dvars_
    from ..lib.metrics import bold_mean_std
    import pandas as pd
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.mean(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]
    dvars_df = pd.DataFrame()
    dvars_df['DVARs'] = dvars_(data_masked)
    mean, std = bold_mean_std(data_masked)
    dvars_df['BOLD_Mean'] = mean
    dvars_df['BOLD_STD'] = std
    return dvars_df


def reho(nib_img, nn=3,
         verbose: bool = True,
         io_handler: Optional[IO] = None):
    from ..lib.metrics import regional_homogeneity
    data, affine, resol = decomp_dataobj(nib_img)
    data_reho = regional_homogeneity(data, nn=nn)
    return save_to_nib(data_reho, affine)


def alff(nib_img, dt, lowcut, highcut,
         verbose: bool = True,
         io_handler: Optional[IO] = None):
    from ..lib.metrics import amplitude_low_freq_fluctuation
    data, affine, resol = decomp_dataobj(nib_img)
    mask_idx = np.nonzero(data.mean(-1))
    data_masked = data[mask_idx]
    alff_masked = amplitude_low_freq_fluctuation(data_masked, dt=dt, lowcut=lowcut, highcut=highcut)
    data_alff = np.zeros(data.shape[:3])
    data_alff[mask_idx] = alff_masked
    return save_to_nib(data_alff, affine)


def tsnr(nib_img,
         verbose: bool = True,
         io_handler: Optional[IO] = None):
    from ..lib.filter.signal import tsnr
    data, affine, resol = decomp_dataobj(nib_img)
    mask_idx = np.nonzero(data.mean(-1))
    data_masked = data[mask_idx]
    tsnr_masked = tsnr(data_masked)
    data_tsnr = np.zeros(data.shape[:3])
    data_tsnr[mask_idx] = tsnr_masked
    return save_to_nib(data_tsnr, affine)


def nuisance_regression(nib_img, mask: Optional[nib.Nifti1Image] = None,
                        port: int =1,
                        ort: Optional[np.ndarray] = None,
                        dt: Union[int, float] = 1.0,
                        lowcut: Union[float] = 0.01,
                        highcut: Union[float] = 0.1,
                        fwhm: Optional[float] = None,
                        verbose: bool = True,
                        io_handler: Optional[IO] = None):
    from ..lib.filter import bandpass, linear_regression, gaussian_smoothing, polynomial_feature
    from ..lib.io import nib2sitk, sitk2nib

    if io_handler is None:
        from sys import stdout
        io_handler = stdout
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.mean(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]

    # detrend data
    model = polynomial_feature(data_masked, order=port)
    if ort is not None:
        model = np.concatenate([model, ort], axis=-1)
    fit_masked = linear_regression(data_masked, model=model)
    data_masked -= fit_masked.sum(-1)

    # bandpass filter
    data_masked = bandpass(data_masked, dt=dt, lowcut=lowcut, highcut=highcut)
    data_masked += fit_masked[..., 0]
    data[mask_idx] = data_masked
    filtered_nii = save_to_nib(data, affine)

    # gaussian smoothing
    if fwhm is not None:
        sitk_img, header = nib2sitk(filtered_nii)
        filtered_sitk_img = gaussian_smoothing(sitk_img, fwhm, io_handler=io_handler)
        filtered_nii = sitk2nib(filtered_sitk_img, header)
    return filtered_nii


def modenorm(nib_img, mode=1000):
    from ..lib.filter.signal import mode_normalization
    data, affine, resol = decomp_dataobj(nib_img)
    mask_idx = np.nonzero(data.mean(-1))
    data_masked = data[mask_idx]
    mn_masked = mode_normalization(data_masked, mode=mode)
    data_mn = np.zeros(data.shape)
    data_mn[mask_idx] = mn_masked
    return save_to_nib(data_mn, affine)


def censor_ts(nib_img, censored_mask):
    data, affine, resol = decomp_dataobj(nib_img)
    data_censored = data[..., censored_mask]
    return save_to_nib(data_censored, affine)