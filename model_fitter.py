import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


class ModelFitter:
    """A class that performs actual model fitting via statsmodels.
    See my master's thesis (@pxie2016/UWThesis),
    Muggeo (2003), or Muggeo et al (2014) for details.
    """

    def __init__(self, conv_depth: int = 20) -> None:
        # TODO: Iterative model fitting with bootstrap restarting
        pass
