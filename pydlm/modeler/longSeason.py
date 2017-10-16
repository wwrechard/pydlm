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
from numpy import matrix
from .dynamic import dynamic


class longSeason(dynamic):
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
                 data=None,
                 period=4,
                 stay=7,
                 discount=0.99,
                 name='longSeason',
                 w=100):

        self.period = period
        self.stay = stay
        if data is None:
            raise NameError('Data must be provided for longSeason.')

        # create features. nextState and state are used to
        # remember the next feature shap
        features, self.nextState = self.createFeatureMatrix(period=period,
                                                            stay=stay,
                                                            n=len(data),
                                                            state=[0, 0])

        dynamic.__init__(self,
                         features=features,
                         discount=discount,
                         name=name,
                         w=w)
        self.checkDataLength()

        # modify the type to be autoReg
        self.componentType = 'longSeason'

    def createFeatureMatrix(self, period, stay, n, state):
        """ Create the feature matrix based on the supplied data and the degree.

        Args:
            period: the periodicity of the component
            stay: the length of the base unit, i.e, how long before change to
                  change to the next state.
        """

        # initialize feature matrix
        nextState = state
        features = []

        count = 0
        while count < n:
            # create new feature for next state
            new_feature = [0] * period
            new_feature[nextState[0]] = 1
            features.append(new_feature)

            # update the state
            nextState[1] = (nextState[1] + 1) % stay
            if nextState[1] == 0:
                nextState[0] = (nextState[0] + 1) % period

            count += 1

        return features, nextState

    # the degree cannot be longer than data
    def checkDataLength(self):
        """ Check whether the degree is less than the time series length

        """
        if self.d >= self.n:
            raise NameError('The degree cannot be longer than the data series')

    # override
    def appendNewData(self, newData):
        """ Append new data to the existing features. Overriding the same method in
        @dynamic

        Args:
            newData: a list of new data

        """
        # create the new features
        incrementLength = len(newData) + self.n - len(self.features)
        if incrementLength > 0:
            newFeatures, \
                self.nextState = self.createFeatureMatrix(period=self.period,
                                                          stay=self.stay,
                                                          n=incrementLength,
                                                          state=self.nextState)
        self.features.extend(newFeatures)
        self.n += len(newData)

    # override
    def popout(self, date):
        """ Pop out the data of a specific date and rewrite the correct feature matrix.

        Args:
            date: the index of which to be deleted.

        """
        # Since the seasonality is a fixed patten,
        # no matter what date is popped out
        # we just need to remove the last date,
        # otherwise the feature patten will be
        # changed.

        # if you want to delete a date and change
        # the underlying patten, i.e., shorten
        # the periodicity of the period that date
        # is presented, you should use ignore
        # instead
        print('Popout the date will change the whole' +
              ' seasonality patten on all the' +
              'future days. If you want to keep the' +
              ' seasonality patten on the future' +
              'days unchanged. Please use ignore instead')

        self.features.pop()
        self.n -= 1

        # push currentState back by 1 day. Need to take care of all
        # corner cases.
        if self.nextState[1] == 0:
            self.nextState[1] = self.stay - 1
            if self.nextState[0] == 0:
                self.nextState[0] = self.period - 1
            else:
                self.nextState[0] -= 1
        else:
            self.nextState[1] -= 1

    def alter(self, date, dataPoint):
        """ We do nothing to longSeason, when altering the main data

        """

        # do nothing
        pass

    def updateEvaluation(self, step):
        """ update the evaluation matrix to a specific date
        This function is used when fitting the forward filter and
        backward smoother
        in need of updating the correct evaluation matrix

        """
        if step < len(self.features):
            self.evaluation = matrix(self.features[step])
        else:
            newFeatures, \
                self.nextState = self.createFeatureMatrix(
                    period=self.period,
                    stay=self.stay,
                    n=step + 1 - len(self.features),
                    state=self.nextState)
            self.features.extend(newFeatures)
            self.evaluation = matrix(self.features[step])
        self.step = step
