from changepoint import Changepoint
from data_simulator import DataSimulator
from model_fitter import ModelFitter
from model_visualizer import ModelVisualizer
from sklearn import datasets
import pandas as pd


class ControlPanel:
    """
    A class that coordinates all other modules to fit any and all models, and
    accepts user-supplied parameters. The "C" in MVC.
    """

    def __init__(self) -> None:
        # The omnipresent, quintessential, and slightly cliche-y dataset, iris
        iris = datasets.load_iris()
        self.df = pd.DataFrame(iris.data[:, :2])
        self.ds, self.mf, self.mv = None, None, None
        self.est_cp = None
        # Some default parameters for the data simulator, if it is ever used
        self.ds_params = {"sample_size": 500, "lb": 0, "ub": 1, "intercept": 0, "init_slope": 0.5,
                          "slope_change": 2, "noise_sd": 0.1}
        self.ds_params["true_cp"] = Changepoint.fixed(0.5, self.ds_params["sample_size"])
        self.mf_params = {"conv_depth": 20, "br": False,
                          "curr_cp": Changepoint.fixed(0.9, self.ds_params["sample_size"]),
                          "cov_specific_cp": False, "cov_specific_slope": False}

    def init_ds(self) -> None:
        self.ds = DataSimulator(self.ds_params)
        self._df = self.ds.get_df()

    def init_mf(self) -> None:
        self.mf = ModelFitter(self._df, self.mf_params)

    def fit(self) -> None:
        self.mf.fit()
        self.est_cp = self.mf.get_res()

    def plot(self) -> None:
        self.mv = ModelVisualizer(self._df, self.est_cp)
        self.mv.plot()

    def print_df(self) -> None:
        print(self.df)
