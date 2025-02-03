import numpy as np
import pandas as pd


def linear_interpolate_missing_vals(arr):
    """
    Interpolates missing values in a 2D numpy array using linear interpolation along each column.

    The function first addresses the initial and final row NaN values by substituting them with the first and last non-NaN values found in each column, respectively. It then uses linear interpolation to estimate and fill in all remaining NaN values throughout the array.

    Parameters
    ----------
    arr : np.ndarray
            A 2D numpy array of shape (n_samples, n_features) with missing values (NaNs) that need to be interpolated.

    Returns
    -------
    np.ndarray
            A 2D numpy array of the same shape as the input, where all NaN values have been filled in using linear interpolation.

    Examples
    --------
    >>> arr = np.array([[np.nan, 2, np.nan], [np.nan, np.nan, np.nan], [7, np.nan, 9], [np.nan, 10, np.nan]])
    >>> linear_interpolate_missing_vals(arr)
    array([[ 7.,  2.,  9.],
               [ 7.,  6.,  9.],
               [ 7., 10.,  9.],
               [ 7., 10.,  9.]])

    """

    B, D = arr.shape
    head_not_nan_ind = (~np.isnan(arr)).argmin(axis=0)  # D
    tail_not_nan_ind = (~np.isnan(arr[::-1])).argmin(axis=0)  # D
    tail_not_nan_ind = B - 1 - tail_not_nan_ind

    arr[0, :] = arr[head_not_nan_ind, np.arange(D)]
    arr[-1, :] = arr[tail_not_nan_ind, np.arange(D)]

    df = pd.DataFrame(arr)

    res = df.interpolate(method="linear", axis=0, limit_direction="both")

    return res.to_numpy()


def spline_interpolate_missing_vals(arr):
    """
    Interpolates missing values in a numpy array using spline interpolation.

    Parameters
    ----------
    arr : np.ndarray
            A 2D numpy array of shape (n_samples, n_features) with missing values (NaNs).

    Returns
    -------
    np.ndarray
            A 2D numpy array of the same shape as the input, where all NaN values have been filled in using spline interpolation.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[np.nan, 2, np.nan], [np.nan, np.nan, np.nan], [7, np.nan, 9], [np.nan, 10, np.nan]])
    >>> spline_interpolate_missing_vals(arr)
    array([[ 7.,  2.,  9.],
               [ 7.,  6.,  9.],
               [ 7., 10.,  9.],
               [ 7., 10.,  9.]])

    """
    B, D = arr.shape
    head_not_nan_ind = (~np.isnan(arr)).argmin(axis=0)  # D
    tail_not_nan_ind = (~np.isnan(arr[::-1])).argmin(axis=0)  # D
    tail_not_nan_ind = B - 1 - tail_not_nan_ind

    arr[0, :] = arr[head_not_nan_ind, np.arange(D)]
    arr[-1, :] = arr[tail_not_nan_ind, np.arange(D)]

    df = pd.DataFrame(arr)

    res = df.interpolate(method="polynomial", axis=3, limit_direction="both")

    return res.to_numpy()
