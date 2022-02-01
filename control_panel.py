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
        # The omnipresent, quintessential, and slightly cliche-y default dataset, iris
        iris = datasets.load_iris()
        self.df = pd.DataFrame(iris.data[:, :2])
        self.ds, self.mf, self.mv = None, None, None
        self.est_cp = None
        # Some default parameters for the data simulator, if it is ever used
        self.ds_params = {"sample_size": 5000, "lb": 0, "ub": 1, "intercept": 0, "init_slope": 0.5,
                          "slope_change": 2, "noise_sd": 0.75}
        self.ds_params["true_cp"] = Changepoint.fixed(0.5, self.ds_params["sample_size"])
        self.mf_params = {"conv_depth": 20, "br": True,
                          "curr_cp": Changepoint.fixed(0.9, self.ds_params["sample_size"]),
                          "cov_specific_cp": False, "cov_specific_slope": False}

    """
    Allows user to set up their own data simulation parameters; merely a setter of self.ds_params for now,
    but will be more logically sound and user-friendly later
    """

    def simulator_setup(self, ds_params: dict) -> None:
        self.ds_params = ds_params

    """
    Allows user to set up their own model parameters; merely a setter of self.mf_params for now,
    but will be more logically sound and user-friendly later
    """

    def model_setup(self, mf_params: dict) -> None:
        self.mf_params = mf_params

    def init_ds(self) -> None:
        self.ds = DataSimulator(self.ds_params)
        self.df = self.ds.get_df()

    def init_mf(self) -> None:
        self.mf = ModelFitter(self.df, self.mf_params)

    def fit(self) -> None:
        self.mf.fit()
        self.est_cp = self.mf.get_res()

    def plot(self) -> None:
        self.mv = ModelVisualizer(self.df, self.est_cp)
        self.mv.plot()

    def print_df(self) -> None:
        print(self.df)
