from changepoint import Changepoint
from data_simulator import DataSimulator
from model_fitter import ModelFitter
from model_visualizer import ModelVisualizer

"""Generate sim"""
ds = DataSimulator(sample_size=1000, lb=0, ub=1, intercept=0, init_slope=0.5, slope_change=2,
                   true_cp=Changepoint.fixed(0.5, 1000), noise_sd=0.1)
mv = ModelVisualizer(ds.get_df())
mv.plot()


