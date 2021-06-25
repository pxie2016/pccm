import pandas as pd
import statsmodels.formula.api as smf
from changepoint import Changepoint


class ModelFitter:
    """A class that performs actual model fitting via statsmodels.
    See my master's thesis (@pxie2016/UWThesis),
    Muggeo (2003), or Muggeo et al (2014) for details.
    """

    def __init__(self, data: pd.DataFrame, curr_cp: Changepoint, conv_depth: int = 20) -> None:
        # TODO: Iterative model fitting with bootstrap restarting
        self.data = data
        self.curr_cp = curr_cp.value
        self.conv_depth = conv_depth

    def one_rep(self) -> None:
        data_copy = self.data.copy().assign(curr_cp=self.curr_cp)
        ind = (data_copy.x > data_copy.curr_cp).astype(int)
        data_copy = data_copy.assign(u_tilde=ind * (data_copy.x - data_copy.curr_cp),
                                     v_tilde=-ind)
        print(data_copy)
        mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=data_copy)
        res = mod.fit()
        print(res.summary())
        self.update()

    def update(self) -> None:
        pass

    def fit(self) -> None:
        for _ in range(self.conv_depth):
            self.one_rep()
