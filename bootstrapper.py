import pandas as pd


class Bootstrapper:
    """
    A class that resamples the dataset provided with replacement for further use
    in the ModelFitter class. Produces a list of pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame, num_copies: int = 10) -> None:
        self.df = df
        self.num_copies = num_copies
        self.list_of_dfs = [self.boot_one() for _ in range(num_copies)]

    def boot_one(self):
        return self.df.sample(frac=1, replace=True)
