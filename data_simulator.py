import numpy as np
import pandas as pd
from numpy.random import default_rng


class DataSimulator:
    """
    A class that generates simulated data with a single breakpoint
    for testing and demonstration purposes.
    """

    def __init__(self, ds_params: dict) -> None:
        rng = default_rng()
        self._x = rng.uniform(low=ds_params["lb"], high=ds_params["ub"], size=ds_params["sample_size"])
        self._noise = rng.normal(loc=0, scale=ds_params["noise_sd"], size=ds_params["sample_size"])
        self._true_cp = ds_params["true_cp"].value
        self._y = [ds_params["intercept"]] * ds_params["sample_size"] + ds_params["init_slope"] * self._x + \
            ds_params["slope_change"] * (self._x > self._true_cp) * (self._x - self._true_cp) + self._noise
        # Collate into a pandas DataFrame only when needed
        self._df = pd.DataFrame(data=np.column_stack((self._x, self._y)),
                                columns=['x', 'y'])

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._df)
