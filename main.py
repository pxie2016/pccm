from changepoint import Changepoint
from data_simulator import DataSimulator
from model_fitter import ModelFitter
from model_visualizer import ModelVisualizer

# Generate some simulated data for visualization
ds = DataSimulator(sample_size=1000, lb=0, ub=1, intercept=0, init_slope=0.5, slope_change=2,
                   true_cp=Changepoint.fixed(0.5, 1000),
                   noise_sd=0.1)

# Plot the dataset
mv = ModelVisualizer(ds.get_df())
mv.plot()

# Fit the model
mf = ModelFitter(ds.get_df(), curr_cp=Changepoint.fixed(0.55, 1000))
mf.one_rep()

