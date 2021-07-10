from bootstrapper import Bootstrapper
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.regression.linear_model as sm_ols


class ModelFitter:
    """A class that performs actual model fitting via statsmodels.
    See my master's thesis (@pxie2016/UWThesis),
    Muggeo (2003), or Muggeo et al (2014) for details.
    """

    def __init__(self, df: pd.DataFrame, mf_params: dict) -> None:
        # TODO: Iterative model fitting with bootstrap restarting (S. Wood, 2001)
        # Time complexity & relationship with SGD family of minimization algorithms to be explored...
        self.df = df
        self.curr_cp = mf_params["curr_cp"].value
        self.df = self.df.assign(curr_cp=self.curr_cp)
        self.br = mf_params["br"]
        self.conv_depth = mf_params["conv_depth"]
        self.est_cp = None

    def one_rep(self) -> None:
        self.df = self.calc_pre_regression_pseudocovs(self.df)
        mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=self.df)
        res = mod.fit()
        self.df = self.calc_post_regression_pseudocovs(self.df, res)
        # print(res.summary())

    def one_rep_br(self) -> None:
        bs = Bootstrapper(self.df)
        df_copies = [self.df.copy()] * bs.num_copies
        for boot_df in bs.list_of_dfs:
            i = 0
            boot_df = self.calc_pre_regression_pseudocovs(boot_df, True)
            boot_mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=boot_df)
            boot_res = boot_mod.fit()
            boot_df = self.calc_post_regression_pseudocovs(boot_df, boot_res, True)
            df_copies[i] = df_copies[i].assign(boot_cp=boot_df["curr_cp"])
            df_copies[i] = self.calc_pre_regression_pseudocovs(df_copies[i])
            refit_mod = smf.ols(formula='y~x + u_tilde_boot + v_tilde_boot', data=df_copies[i])
            df_copies[i] = self.calc_post_regression_pseudocovs(df_copies[i], refit_mod, True)
            # TODO: df_copies[i] needs to remember their log-likelihood values - a wrapper maybe?

    def fit(self) -> None:
        for _ in range(self.conv_depth):
            self.one_rep_br() if self.br else self.one_rep()
        self.est_cp = self.df["curr_cp"]

    def get_res(self) -> pd.Series:
        return self.est_cp

    """Helpers below"""

    @staticmethod
    def calc_pre_regression_pseudocovs(df: pd.DataFrame, boot: bool = False) -> pd.DataFrame:
        if not boot:
            df = df.assign(ind=(df.x > df.curr_cp).astype(int))
            df = df.assign(u_tilde=df.ind * (df.x - df.curr_cp), v_tilde=-df.ind)
        else:
            df = df.assign(ind_boot=(df.x > df.boot_cp).astype(int))
            df = df.assign(u_tilde_boot=df.ind_boot * (df.x - df.boot_cp), v_tilde_boot=-df.ind_boot)
        return df

    @staticmethod
    def calc_post_regression_pseudocovs(df: pd.DataFrame, reg_result: sm_ols.RegressionResults,
                                        boot: bool = False) -> pd.DataFrame:
        if not boot:
            df = df.assign(u_tilde=reg_result.params["u_tilde"])
            df.curr_cp += reg_result.params["v_tilde"] / df["u_tilde"]
        else:
            df = df.assign(u_tilde_boot=reg_result.params["u_tilde_boot"])
            df.curr_cp += reg_result.params["v_tilde_boot"] / df["u_tilde_boot"]
        return df
