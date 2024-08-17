"""
===========================================================================

The code for long seasonality components

===========================================================================

This code implements the long seasonality component as a sub-class of dynamic.
The difference between the long seasonality is that
1) The seasonality component use each date as a unit and change in a given
periodicy. For example, 1, 2, 3, 4, 1, 2, 3, 4.
2) However, the long seasonality is capable to group a couple of days as the
basic unit and change in a periodicy. For example, 1, 1, 1, 2, 2, 2, 3, 3, 3,
4, 4, 4.
The usecase for long seasonality is to model longer seasonality with the
short-term seasonality. For example, the short-term seansonality can be
used to model the day of a weak patten and the long seasonality can be used
to model the week of a month patten in the same model.
Different from the dynamic component, the features in the autoReg is generated
from the data, and updated according to the data. All other features are
similar to @dynamic.

"""
from .autoReg import autoReg
import logging
import numpy as np


class longSeason(autoReg):
    """ The longSeason class alows user to add a long seasonality component
    to the dlm. The difference between the long seasonality is that
    1) The seasonality component use each date as a unit and change in a given
    periodicity. For example, 1, 2, 3, 4, 1, 2, 3, 4.
    2) However, the long seasonality is capable to group couple of days as the
    basic unit and change in a periodicity. For example, 1, 1, 1, 2, 2, 2,
    3, 3, 3, 4, 4, 4.
    The usecase for long seasonality is to model longer seasonality with the
    short-term seasonality. For example, the short-term seansonality can be
    used to model the day of a weak patten and the long seasonality can be used
    to model the week of a month patten in the same model.
    This code implements the longSeason component as a sub-class of
    dynamic. Different from the dynamic component, the features in the
    autoReg is generated from the data, and updated according to the data.
    All other features are similar to @dynamic.

    Args:
        data: the time series data
        period: the periodicy of the longSeason component
        stay: the length of each state lasts
        discount: the discount factor
        name: the name of the trend component
        w: the value to set the prior covariance. Default to a diagonal
           matrix with 1e7 on the diagonal.

    Attributes:
        period: the periodicity, i.e., how many different states it has in
                one period
        stay: the length of a state last.
        discount factor: the discounting factor
        name: the name of the component

    """


    def __init__(self,
                 data=None,  # DEPRECATED
                 period=4,
                 stay=7,
                 discount=0.99,
                 name='longSeason',
                 w=100):

        self.period = period
        self.stay = stay
        if data is not None:
            logging.warning('The data argument in longSeason is deprecated. Please avoid using it.')

        super().__init__(data=data,
                         degree=period,
                         discount=discount,
                         name=name,
                         w=w)

        # modify the type to be longSeason
        self.componentType = 'longSeason'

        # Initialize the evaluation vector
        self.evaluation = np.matrix([0] * self.period)


    def updateEvaluation(self, step, data=None):
        """ update the evaluation matrix to a specific date
        This function is used when fitting the forward filter and
        backward smoother
        in need of updating the correct evaluation matrix

        """
        # Calculate the right position for value 1
        position = int(step / self.stay) % self.period
        self.evaluation[0, position] = 1
        self.evaluation[0, position - 1] = 0
