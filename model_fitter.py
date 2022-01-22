import bootstrapper
from bootstrapper import Bootstrapper
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.regression.linear_model as sm_ols


class ModelFitter:
    """A class that performs actual model fitting via statsmodels.
    See my master's thesis (@pxie2016/UWThesis),
    Muggeo (2003), or Muggeo et al (2014) for details.
    """

    def __init__(self, df: pd.DataFrame, mf_params: dict) -> None:
        self.df = df
        self.curr_cp = mf_params["curr_cp"].value
        self.df = self.df.assign(curr_cp=self.curr_cp)
        self.br = mf_params["br"]
        self.conv_depth = mf_params["conv_depth"]
        self.est_cp = None
        self.res = None

    def one_rep(self) -> None:
        self.df = self.calc_pre_regression_pseudocovs(self.df)
        mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=self.df)
        self.res = mod.fit()
        self.df = self.calc_post_regression_pseudocovs(self.df, self.res)
        # print(res.summary())

    def one_rep_bootstrap(self) -> None:
        self.one_rep()
        loglik = self.res.llf

        bs = Bootstrapper(self.df)
        df_copies = [self.df.copy()] * bs.num_copies
        loglik_copies = [0.] * bs.num_copies
        self.bootstrap_and_refit(bs, df_copies, loglik_copies)

        if loglik < max(loglik_copies):
            max_index = np.argmax(loglik_copies)
            self.df = df_copies[max_index]

    def fit(self) -> None:
        for _ in range(self.conv_depth):
            self.one_rep_bootstrap() if self.br else self.one_rep()
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
            df = df.assign(ind_boot=(df.x > df.curr_cp).astype(int))
            df = df.assign(u_tilde_boot=df.ind_boot * (df.x - df.curr_cp), v_tilde_boot=-df.ind_boot)
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

    def bootstrap_and_refit(self, bs: Bootstrapper, df_copies: [pd.DataFrame], loglik_copies: [float]) -> None:
        for i in range(bs.num_copies):
            boot_df = df_copies[i]
            boot_df = boot_df.assign(curr_cp=self.curr_cp)
            boot_df = self.calc_pre_regression_pseudocovs(boot_df)
            boot_mod = smf.ols(formula='y~x + u_tilde + v_tilde', data=boot_df)
            boot_res = boot_mod.fit()
            boot_df = self.calc_post_regression_pseudocovs(boot_df, boot_res)
            df_copies[i] = df_copies[i].assign(curr_cp=boot_df["curr_cp"])
            df_copies[i] = self.calc_pre_regression_pseudocovs(df_copies[i], True)
            refit_mod = smf.ols(formula='y~x + u_tilde_boot + v_tilde_boot', data=df_copies[i])
            refit_mod_res = refit_mod.fit()
            df_copies[i] = self.calc_post_regression_pseudocovs(df_copies[i], refit_mod_res, True)
            loglik_copies[i] = refit_mod_res.llf