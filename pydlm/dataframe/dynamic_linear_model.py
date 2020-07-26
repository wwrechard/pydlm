"""
currently it's only a wrapper of the dlm class
"""
from pydlm.dataframe.utils import get_time_series_from_dataframe
from pydlm.dlm import dlm
from pydlm.modeler.dynamic import dynamic


class DynamicLinearModel(dlm):
    """ This class lets you specify dlm by providing a pandas dataframe

    It currently wraps the main dlm class and only acts as a helper class

    Example 1:
        >>> # randomly generate fake data on 1000 days
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pydlm.dataframe.dynamic_linear_model import DynamicLinearModel as dlm
        >>> from pydlm import trend, seasonality
        >>> data = pd.DataFrame({'y':np.random.random((1, 1000))})
        >>> # construct the dlm of a linear trend and a 7-day seasonality
        >>> myDlm = dlm(data, target_col='y') + trend(2, 0.98) + seasonality(7, 0.98)
        >>> # filter the result
        >>> myDlm.fitForwardFilter()
        >>> # extract the filtered result
        >>> myDlm.getFilteredObs()

    Example 2 (fit a linear regression):
        >>> from pydlm import dynamic
        >>> data = np.random.random((1, 100))
        >>> myDlm = dlm(data) + trend(1, 0.98, name='a') + dynamic([[i] for i in range(100)], 1, name='b')
        >>> myDlm.fit()
        >>> coef_a = myDlm.getLatentState('a')
        >>> coef_b = myDlm.getLatentState('b')

    Attributes:
       data: a list of doubles of the raw time series data.
             It could be either the python's built-in list of
             doubles or numpy 1d array.

    """

    def __init__(self, data, target_col,
                 time_step_col=None, x_cols=None,
                 dynamic_component_name=None,
                 dynamic_component_discount=None,
                 dynamic_component_w=None):

        if time_step_col is not None:
            data = data.sort_values(time_step_col)

        time_series = get_time_series_from_dataframe(data, target_col, time_step_col)
        super(DynamicLinearModel, self).__init__(data=time_series)

        if x_cols is not None:
            features = data.loc[:, x_cols].values
            _component_name = dynamic_component_name if dynamic_component_name else 'dynamic'
            _component_discount = dynamic_component_discount if dynamic_component_discount else 0.99
            _component_w = dynamic_component_w if dynamic_component_w else 100
            dynamic_component = dynamic(features=features,
                                        name=_component_name,
                                        discount=_component_discount,
                                        w=_component_w)

            self.__add__(dynamic_component)
