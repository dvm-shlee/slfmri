from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from .norm import standardize
from ..utils import iszero
from ..errors import *


def linear_regression(signal, estimator, design_matrix):
    if iszero(signal):
        return signal
    else:
        # Applying regression de-noising
        model = estimator()
        model.fit(design_matrix, signal)
        return model


def polynomial_fit(signal: np.ndarray, estimator, order=3):
    """ Estimate polynomial curve fit for data
    Args:
        signal: time series data
        estimator: estimator for linear regression
        order: order of polynomial curve
    Returns
        fitted signal
    """
    model = make_pipeline(PolynomialFeatures(order), estimator())
    x = np.linspace(0, (len(signal) - 1) * 2, len(signal))
    x_ = x[:, np.newaxis]
    model.fit(x_, signal)
    return np.asarray(model.predict(x_))


def nuisance_regression(signal, estimator, ort=None, order=3):
    polort = pd.DataFrame(polynomial_fit(signal, estimator, order=order))

    if ort is None:
        design_matrix = polort
    else:
        if isinstance(ort, list):
            ort_list = [polort]
            for o in ort:
                o = pd.DataFrame(o)
                ort_list.extend([o, o.diff().fillna(0)])
            design_matrix = pd.concat(ort_list, axis=1, ignore_index=True)
        else:
            ort = pd.DataFrame(ort)
            design_matrix = pd.concat([polort, ort, ort.diff().fillna(0)],
                                      axis=1, ignore_index=True)
    design_matrix = standardize(design_matrix, axis=0)

    model = linear_regression(signal, estimator, design_matrix)
    if isinstance(model, np.ndarray):
        return model
    else:
        regressor = model.predict(design_matrix)
        regressor -= regressor.mean()
        return np.asarray(signal - regressor)
