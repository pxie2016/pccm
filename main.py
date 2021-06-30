from changepoint import Changepoint
from data_simulator import DataSimulator
from model_fitter import ModelFitter
from model_visualizer import ModelVisualizer

SAMPLE_SIZE = 100

# Generate some simulated data for visualization
ds = DataSimulator(sample_size=SAMPLE_SIZE, lb=0, ub=1, intercept=0, init_slope=0.5, slope_change=2,
                   true_cp=Changepoint.fixed(0.5, SAMPLE_SIZE),
                   noise_sd=0.1)

# Fit the model
mf = ModelFitter(ds.get_df(), curr_cp=Changepoint.fixed(0.9, SAMPLE_SIZE))
mf.fit()

# Plot the results
mv = ModelVisualizer(mf)
mv.plot()
