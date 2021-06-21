import numpy as np
import pandas as pd
from changepoint import Changepoint
from numpy.random import default_rng


class DataSimulator:
    """
    A class that generates simulated data with a single breakpoint
    for testing and demonstration purposes.
    """

    def __init__(self, sample_size: int, lb: float, ub: float, intercept: float, init_slope: float,
                 slope_change: float, noise_sd: float, true_cp: Changepoint) -> None:
        rng = default_rng()
        self._lb = lb
        self._ub = ub
        self._sample_size = sample_size
        self._x = rng.uniform(low=lb, high=ub, size=sample_size)
        self._noise = rng.normal(loc=0, scale=noise_sd, size=sample_size)
        self._curr_cp = [0] * sample_size
        self._true_cp = true_cp.value
        self._y = [intercept] * sample_size + init_slope * self._x + \
            slope_change * (self._x > self._true_cp) * (self._x - self._true_cp) + self._noise
        self._df = pd.DataFrame(data=np.column_stack((self._x, self._y)), columns=['x', 'y'])

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._df)
