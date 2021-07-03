import pandas as pd
import statsmodels.formula.api as smf


class ModelFitter:
    """A class that performs actual model fitting via statsmodels.
    See my master's thesis (@pxie2016/UWThesis),
    Muggeo (2003), or Muggeo et al (2014) for details.
    """

    def __init__(self, data: pd.DataFrame, mf_params: dict) -> None:
        # TODO: Iterative model fitting with bootstrap restarting (S. Wood, 2001)
        # Time complexity & relationship with SGD family of minimization algorithms to be explored...
        self.data = data
        self.curr_cp = mf_params["curr_cp"].value
        self.br = mf_params["br"]
        self.conv_depth = mf_params["conv_depth"]
        self.est_cp = None

    def one_rep(self) -> None:
        self.data = self.data.assign(curr_cp=self.curr_cp)
        ind = (self.data.x > self.data.curr_cp).astype(int)
        self.data = self.data.assign(u_tilde=ind * (self.data.x - self.data.curr_cp),
                                     v_tilde=-ind)
        print(self.data)
        mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=self.data)
        res = mod.fit()
        self.data = self.data.assign(u_tilde=res.params["u_tilde"])
        self.curr_cp += res.params["v_tilde"]/self.data["u_tilde"]
        # print(res.summary())

    def fit(self) -> None:
        for _ in range(self.conv_depth):
            self.one_rep()
        self.est_cp = self.data["curr_cp"]

    def get_res(self) -> pd.Series:
        return self.est_cp
