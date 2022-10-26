from src.utils.miscs import config_sys_path
config_sys_path(".")

import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, DotProduct, WhiteKernel
from sklearn.exceptions import ConvergenceWarning


class GaussianProcessImputer:
    """Time series imputer using Gaussian Process.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel, default=None
      The kernel specifying the covariance function of the GP.

    periodicity : int, default=95
      The periodicity of the time series.

    batch_size : int, default=200
      The batch size for time series segmentation.

    bin_size : int, default=None
      The bin size for discretizing the time series.

    random_state : int, default=0
      The random state for the GP.

    alpha : float, default=1e-10
      The jitter added to the diagonal of the kernel matrix during fitting.

    normalize_y : bool, default=True
      Whether to normalize the time series.

    n_jobs : int, default=4
      The number of jobs for parallel processing.

    Attributes
    ----------
    gp : sklearn.gaussian_process.GaussianProcessRegressor
      The Gaussian Process regressor.

    Examples
    --------
    >>> import numpy as np
    >>> from src.imputation.gp.gp_imputer import GaussianProcessImputer
    >>> imputer = GaussianProcessImputer(bin_size=1, batch_size=10)
    >>> data = np.array([[1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
    >>>                  [1, 2, 3, 4, 5, 6, np.nan, np.nan, 9, 10]])
    >>> X_impute, y_impute = imputer.impute(data)
    """

    def __init__(
        self,
        kernel=None,
        periodicity=95,
        batch_size=200,
        bin_size=None,
        random_state=0,
        alpha=1e-10,
        normalize_y=True,
        n_jobs=4,
    ):
        self.periodicity = periodicity
        if kernel is None:
            kernel = (
                ExpSineSquared(
                    periodicity=periodicity,
                    periodicity_bounds="fixed",
                    length_scale_bounds=(100, 1000),
                )
                * DotProduct()
                + ExpSineSquared(
                    periodicity=periodicity,
                    periodicity_bounds="fixed",
                    length_scale_bounds=(0.5, 1),
                )
                + WhiteKernel(0.01, noise_level_bounds=(1e-3, 1e0))
            )
        self.kernel = kernel
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.random_state = random_state
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_jobs = n_jobs
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )

    def impute(self, data, keep_raw=True, anomaly_threshold=300):
        """Fit the GP imputer and impute the missing values.

        Parameters
        ----------
        data : np.ndarray, shape=(n_samples, n_timesteps)
          The time series data.

        keep_raw : bool, default=True
          Whether to keep the raw data.

        anomaly_threshold : int, default=300
          The threshold for anomaly detection.

        Returns
        -------
        X_impute : np.ndarray, shape=(n_samples, n_timesteps)
          The time axis of the imputed time series data.

        y_impute : np.ndarray, shape=(n_samples, n_timesteps)
          The value axis of the imputed time series data.
        """

        ymax = np.nanmax(data)
        digitize_bins = np.arange(ymax // self.bin_size + 1) * self.bin_size

        self.len_time = data.shape[1]
        X_impute = np.broadcast_to(np.arange(self.len_time), data.shape)
        y_impute = np.empty_like(data)
        for i, yi in tqdm(enumerate(data), total=len(data)):
            X_train = np.arange(len(yi))[~np.isnan(yi)].reshape(-1, 1)
            if len(X_train) < self.batch_size:  # node with too few data
                y_impute[i] = np.full(self.len_time, np.nan)
            else:
                y_train = yi[~np.isnan(yi)]
                if np.all(y_train == y_train[0]):  # constant node
                    y_impute[i] = np.full(self.len_time, y_train[0])
                else:  # normal node
                    y_train = np.digitize(y_train, digitize_bins)

                    X_train_split, y_train_split, impute_range = self._split_train(
                        X_train, y_train
                    )

                    res = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._batch_impute)(X_batch, y_batch, impute_batch)
                        for X_batch, y_batch, impute_batch in zip(
                            X_train_split, y_train_split, impute_range
                        )
                    )
                    yi_impute = np.concatenate(res)
                    y_impute[i] = yi_impute * self.bin_size

        if keep_raw:
            y_impute = self.combine_raw_impute(data, y_impute, anomaly_threshold)
        return X_impute, y_impute

    def _batch_impute(self, X_batch, y_batch, impute_batch):
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if len(X_batch) == 0:
            return np.full(impute_batch[1] - impute_batch[0], np.nan)

        gp = deepcopy(self.gp)
        gp.fit(X_batch, y_batch)

        X_impute = np.arange(*impute_batch).reshape(-1, 1)
        y_impute = gp.predict(X_impute)
        assert len(X_impute) == len(y_impute)
        y_impute[y_impute < 0] = 0
        return y_impute.ravel()

    def _split_train(self, X_train, y_train):
        split_idx = []
        impute_range = []
        cum_flag = 0
        slice_start = 0
        impute_start = 0
        last_x = 0
        for i, x in enumerate(X_train.ravel()):
            no_gap = x - last_x <= 2 * self.periodicity
            if no_gap:
                cum_flag += 1
            if cum_flag == self.batch_size:
                split_idx.append(slice(slice_start, i + 1))
                impute_range.append((impute_start, x + 1))
                cum_flag = 0
                slice_start = i + 1
                impute_start = x + 1
            elif ~no_gap:
                # extend previous batch
                if len(split_idx) > 0:
                    if split_idx[-1].start != split_idx[-1].stop:
                        split_idx[-1] = slice(split_idx[-1].start, i)
                    impute_range[-1] = (impute_range[-1][0], last_x + 1)
                    # add nan batch
                    split_idx.append(slice(i + 1, i + 1))
                    impute_range.append((last_x + 1, x + 1))
                else:
                    # add nan batch
                    split_idx.append(slice(i + 1, i + 1))
                    impute_range.append((last_x, x + 1))
                cum_flag = 0
                slice_start = i + 1
                impute_start = x + 1

            last_x = x

        if len(impute_range) == 0:  # no imputable range
            impute_range.append((0, self.len_time))
        else:
            if impute_range[0][0] > 0:  # range not start from 0
                if split_idx[0].start <= self.periodicity:
                    impute_range[0] = (0, impute_range[0][1])
                else:
                    split_idx.insert(0, slice(0, 0))
                    impute_range.insert(0, (0, impute_range[0][0]))
            if impute_range[-1][1] < self.len_time:  # range not end at len_time
                if split_idx[-1].stop >= self.len_time - self.periodicity:
                    impute_range[-1] = (impute_range[-1][0], self.len_time)
                else:
                    split_idx.append(slice(split_idx[-1].stop, split_idx[-1].stop))
                    impute_range.append((impute_range[-1][1], self.len_time))
        X_train_split = [X_train[s] for s in split_idx]
        y_train_split = [y_train[s] for s in split_idx]
        return X_train_split, y_train_split, impute_range

    @staticmethod
    def get_anomaly_idx(x, anomaly_threshold=300):
        x_diff_forward = np.diff(x, prepend=0)
        x_diff_backward = np.diff(x, append=0)
        x_diff = (np.abs(x_diff_forward) + np.abs(x_diff_backward)) / 2
        anomaly_idx = x_diff > anomaly_threshold
        return anomaly_idx & (~np.isnan(x))

    def combine_raw_impute(self, data, yhat, anomaly_threshold=300):
        """Combine the raw data and the imputed data."""
        yfinal = np.empty_like(yhat)
        for i in range(len(data)):
            y_i = data[i]
            yhat_i = yhat[i]
            yfinal[i] = deepcopy(yhat_i)
            yfinal[i, ~np.isnan(y_i)] = y_i[~np.isnan(y_i)]

            anomaly_idx = self.get_anomaly_idx(yfinal[i], anomaly_threshold)
            yfinal[i, anomaly_idx] = yhat_i[anomaly_idx]
        return yfinal
