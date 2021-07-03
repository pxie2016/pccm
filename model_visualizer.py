import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ModelVisualizer:
    """
    A class that visualizes either the simulated data or (real-world) input data,
    with various options to display results from ModelFitter. "V" in MVC.
    """

    def __init__(self, df: pd.DataFrame, est_cp: pd.Series) -> None:
        self.df = df
        self.est_cp = est_cp.mean()  # this should not be mean() in the long run

    def plot(self) -> None:
        sns.set_theme()
        sns.relplot(data=self.df, x='x', y='y')
        plt.axvline(x=self.est_cp, c="red")
        plt.show()
