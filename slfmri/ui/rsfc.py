from ..lib.errors import *
from ..lib.io.orient import decomp_dataobj, save_to_nib, nib
from typing import Optional, Union, IO


class DualReg:
    """
    Dual regression integration using QR decomposition
    """

    def __init__(self, data, model):
        """
        data: VxT matrix where V is the number of voxels, T is the number of timepoints
        model: VxF matrix where V is the number of voxels, F is the number of spatial features
        """
        # private
        self._data = data
        self._model = model
        self._Tt, self._Pt = None, None
        self._Ts, self._Ps = None, None
        self._Coefs, self._Coeft = None, None

    def fit(self, pval=None):
        """
        fit data to model
        """
        self._dual_regression(pval)

    def _dual_regression(self, pval=None):
        """
        Perform spatial regression followed by the temporal regression using QR decomposition
        """
        from slfmri import filter as slftl
        from scipy import stats
        model = self._model

        predicted_t, self._Coeft = slftl.linear_regression(self._data.T, model, method='svd', return_beta=True)
        v, f = model.shape
        dof = v - f
        mse_t = np.square(predicted_t.sum(-1) - self._data.T).sum(-1) / float(dof)
        diag_t = np.linalg.inv(np.dot(model.T, model)).diagonal()
        se_t = np.sqrt((mse_t * np.concatenate([diag_t[:, np.newaxis]], axis=-1)).T)

        t_t = self._Coeft.copy()
        t_t[se_t == 0] = 0
        t_t[np.nonzero(se_t)] /= se_t[np.nonzero(se_t)]

        self._Tt = t_t
        self._Pt = 2 * (1 - stats.t.cdf(abs(t_t), df=dof))

        model = self._Coeft
        predicted_s, self._Coefs = slftl.linear_regression(self._data, model, method='svd', return_beta=True)
        v, f = model.shape
        dof = v - f
        mse_s = np.square(predicted_s.sum(-1) - self._data).sum(-1) / float(dof)
        diag_s = np.linalg.inv(np.dot(model.T, model)).diagonal()
        se_s = np.sqrt((mse_s * np.concatenate([diag_s[:, np.newaxis]], axis=-1)).T)

        t_s = self._Coefs.copy()
        t_s[se_s == 0] = 0
        t_s[np.nonzero(se_s)] /= se_s[np.nonzero(se_s)]

        self._Ts = t_s
        self._Ps = 2 * (1 - stats.t.cdf(abs(t_s), df=dof))
        if pval is not None:
            self._Coeft[self._Pt > pval] = 0
            self._Coefs[self._Ps > pval] = 0

    @property
    def coef_(self):
        """
        Coefficients for each spatial and temporal features
        """
        return dict(spatial=self._Coefs,  # VxF matrix
                    temporal=self._Coeft.T)  # FxT matrix

    @property
    def predicted(self):
        """
        Predicted data from dual regression

        Return:
            np.ndarray (2D): VxT matrix
        """
        return np.dot(self._Coefs, self._Coeft.T)

    @property
    def decoded(self):
        """
        Decoded data using model

        Return:
            np.ndarray (2D): VxT matrix
        """
        return np.dot(self._model, self._Coeft.T)

    @property
    def resid(self):
        """
        The difference between original data and predicted data

        Return:
            np.ndarray (2D): V x T
        """
        return self._data - self.predicted


def dvars(nib_img: nib.Nifti1Image,
          mask: Optional[nib.Nifti1Image] = None,
          verbose: bool = True,
          io_handler: Optional[Union[IO, SilentIO]] = None):
    if io_handler is None:
        if verbose:
            io_handler = sys.stdout
        else:
            io_handler = SilentIO()
    from ..lib.metrics import dvars as dvars_
    from ..lib.metrics import bold_mean_std
    import pandas as pd
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.std(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]
    dvars_df = pd.DataFrame()
    dvars_df['DVARs'] = dvars_(data_masked, io_handler=io_handler)
    mean, std = bold_mean_std(data_masked, io_handler=io_handler)
    dvars_df['BOLD_Mean'] = mean
    dvars_df['BOLD_STD'] = std
    return dvars_df


def reho(nib_img, nn=3, mask_img = None,
         verbose: bool = True,
         io_handler: Optional[Union[IO, SilentIO]] = None):
    if io_handler is None:
        if verbose:
            io_handler = sys.stdout
        else:
            io_handler = SilentIO()
    from ..lib.metrics import regional_homogeneity
    data, affine, resol = decomp_dataobj(nib_img)
    if mask_img is not None:
        mask_data = decomp_dataobj(mask_img)[0]
        data = (data.T * mask_data.T).T
    data_reho = regional_homogeneity(data, nn=nn, io_handler=io_handler)
    return save_to_nib(data_reho, affine)


def alff(nib_img, dt, lowcut, highcut,
         verbose: bool = True,
         io_handler: Optional[IO] = None):
    from ..lib.metrics import amplitude_low_freq_fluctuation
    data, affine, resol = decomp_dataobj(nib_img)
    mask_idx = np.nonzero(data.std(-1))
    data_masked = data[mask_idx]
    alff_masked = amplitude_low_freq_fluctuation(data_masked, dt=dt,
                                                 lowcut=lowcut, highcut=highcut,
                                                 io_handler=io_handler)
    data_alff = np.zeros(data.shape[:3])
    data_alff[mask_idx] = alff_masked
    return save_to_nib(data_alff, affine)


def tsnr(nib_img,
         verbose: bool = True,
         io_handler: Optional[IO] = None):
    from ..lib.metrics.rsfc import tsnr
    data, affine, resol = decomp_dataobj(nib_img)
    mask_idx = np.nonzero(data.std(-1))
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
        mask_idx = np.nonzero(data.std(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]

    # detrend data
    model = polynomial_feature(data_masked, order=port)
    model /= model.max(axis=0, initial=None)
    if ort is not None:
        ort -= ort.mean(0)
        ort /= abs(ort).max(0)
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


def modenorm(nib_img, mode=1000, mask: Optional[nib.Nifti1Image] = None):
    from ..lib.filter.signal import mode_normalization
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.std(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]
    mn_masked = mode_normalization(data_masked, mode=mode)
    data_mn = np.zeros(data.shape)
    data_mn[mask_idx] = mn_masked
    return save_to_nib(data_mn, affine)


def standardize(nib_img, mask: Optional[nib.Nifti1Image] = None):
    from ..lib.filter.signal import standardize
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.std(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]
    st_masked = standardize(data_masked)
    data_st = np.zeros(data.shape)
    data_st[mask_idx] = st_masked
    return save_to_nib(data_st, affine)


def censor_ts(nib_img, censored_mask):
    data, affine, resol = decomp_dataobj(nib_img)
    data_censored = data[..., censored_mask]
    return save_to_nib(data_censored, affine)


def connectivity_strength(nib_img, mask_img = None, corrwith = None, pval=None,
                          verbose: bool = True, pos=False,
                          io_handler: Optional[Union[IO, SilentIO]] = None):
    if io_handler is None:
        if verbose:
            io_handler = sys.stdout
        else:
            io_handler = SilentIO()
    from ..lib.metrics import connectivity_strength
    import pandas as pd
    data, affine, resol = decomp_dataobj(nib_img)
    if mask_img is not None:
        mask_idx = np.nonzero(mask_img.dataobj)
    else:
        mask_idx = np.nonzero(data.std(-1))
    if corrwith is not None:
        y = decomp_dataobj(corrwith)[0][mask_idx]
    else:
        y = None
    masked_cstr = connectivity_strength(data[mask_idx], y=y, pos=pos, pval=pval)

    cstr_map = np.zeros(data.mean(-1).shape)
    cstr_map[mask_idx] = masked_cstr
    return save_to_nib(cstr_map, affine)
