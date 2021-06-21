import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ModelVisualizer:
    """
    A class that visualizes either the simulated data or (real-world) input data,
    with various options to display results from ModelFitter.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def plot(self) -> None:
        sns.set_theme()
        sns.relplot(data=self.df, x='x', y='y')
        plt.show()
