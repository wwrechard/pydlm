"""
for testing purpose
"""

import numpy as np
import pandas as pd

from pydlm import trend, seasonality
from pydlm.dataframe.dynamic_linear_model import DynamicLinearModel as d1
from pydlm.dlm import dlm as d2

if __name__ == '__main__':
    data = pd.DataFrame({'y': np.random.random(1000)})
    # construct the dlm of a linear trend and a 7-day seasonality
    d1m = d1(data, target_col='y') + trend(2, 0.98) + seasonality(7, 0.98)
    d1m.fitForwardFilter()
    d1m.fitBackwardSmoother()

    d2m = d2(data.y.values) + trend(2, 0.98) + seasonality(7, 0.98)
    d2m.fitForwardFilter()
    d2m.fitBackwardSmoother()
