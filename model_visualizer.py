import seaborn as sns
import matplotlib.pyplot as plt

from model_fitter import ModelFitter


class ModelVisualizer:
    """
    A class that visualizes either the simulated data or (real-world) input data,
    with various options to display results from ModelFitter.
    """
    
    def __init__(self, mf: ModelFitter) -> None:
        self.mf = mf

    def plot(self) -> None:
        sns.set_theme()
        sns.relplot(data=self.mf.data, x='x', y='y')
        plt.axvline(x=self.mf.est_cp, c="red")
        plt.show()
