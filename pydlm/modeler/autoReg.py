"""
===========================================================================

The code for autoregressive components

===========================================================================

This code implements the autoregressive component as a sub-class of dynamic.
Different from the dynamic component, the features in the autoReg is generated
from the data, and updated according to the data. All other features are
similar to @dynamic.

"""
from numpy import matrix
from .dynamic import dynamic


class autoReg(dynamic):
    """ The autoReg class allows user to add an autoregressive component to the dlm.
    This code implements the autoregressive component as a sub-class of
    dynamic. Different from the dynamic component, the features in the
    autoReg is generated from the data, and updated according to the data.
    All other features are similar to @dynamic.

    The latent states of autoReg are aligned in the order of
    [today - degree, today - degree + 1, ..., today - 2, today - 1]. Thus,
    when fetching the latents from autoReg component, use this order to
    correctly align the coefficients.

    (TODO: change the implementation of autoReg, so that the component uses
     filteredObs as the feature instead of the observed data. To do this, we
     might need pass the dlm as a parameter to autoReg, so that autoReg can
     fetch the filtered observation on the fly)

    Args:
        data: the time series data
        degree: the order of the autoregressive component
        discount: the discount factor
        name: the name of the trend component
        w: the value to set the prior covariance. Default to a diagonal
           matrix with 1e7 on the diagonal.

    Attributes:
        degree: the degree of autoregressive, i.e., how many days to look back
        data: the time series data used for constructing the autoregressive
              features
        discount factor: the discounting factor
        name: the name of the component
        padding: either 0 or None. The number to be padded for the first degree
                 days, as no previous data is observed to form the feature
                 matrix

    """

    def __init__(self,
                 data=None,
                 degree=2,
                 discount=0.99,
                 name='ar2',
                 w=100,
                 padding=0):

        if data is None:
            raise NameError('data must be provided to construct' +
                            ' autoregressive component')

        # create fake data to incorporate paddings in the beginning
        fakeData = [padding] * degree + [num for num in data]

        # create features
        features = self.createFeatureMatrix(degree=degree, data=fakeData)

        dynamic.__init__(self,
                         features=features,
                         discount=discount,
                         name=name,
                         w=w)
        self.checkDataLength()

        # modify the type to be autoReg
        self.componentType = 'autoReg'

        # the data of the last day has to be stored,
        # since it is not used so far and
        # will be needed when adding new data
        self.lastDay = data[-1]

    def createFeatureMatrix(self, degree, data):
        """ Create the feature matrix based on the supplied data and the degree.

        Args:
            degree: the auto-regressive dependency length.
            data: the raw time series data of the model.
        """

        # we currently don't support missing data for auto regression
        if self.hasMissingData(data):
            raise NameError('The package currently do not support missing ' +
                            'for auto regression. The support has been ' +
                            'implemented, but deprecated due to efficiency ' +
                            'issue. Will support this in next version.')

        # initialize feature matrix
        features = []

        for i in range(degree, len(data)):
            features.append(data[(i - degree):i])

        return features

    # the degree cannot be longer than data
    def checkDataLength(self):
        """ Check whether the degree is less than the time series length

        """
        if self.d >= self.n:
            raise NameError('The degree cannot be longer than the data series')

    # overide
    def updateEvaluation(self, date):
        if date < self.n:
            self.evaluation = matrix(self.features[date])
        elif date == self.n:
            self.evaluation = matrix(self.features[-1][1:] + [self.lastDay])
        else:
            raise NameError('The step is out of range')

    # override
    def appendNewData(self, newData):
        """ Append new data to the existing features. Overriding the same method in
        @dynamic

        Args:
            newData: a list of new data

        """
        # fetch the last entry of the feature
        previousDays = self.features[-1]
        fakeData = [num for num in previousDays] + [self.lastDay] + \
                   [num for num in newData]

        # delete the first day which is duplication
        fakeData.pop(0)

        # using the constructed fake data to create new feature sets
        newFeatures = self.createFeatureMatrix(degree=self.d, data=fakeData)

        # append the new feature to the old ones
        self.features.extend(newFeatures)

        # update the last day
        self.lastDay = newData[-1]

        # update n
        self.n = len(self.features)

    # override
    def popout(self, date):
        """ Pop out the data of a specific date and rewrite the correct feature matrix.

        Args:
            date: the index of which to be deleted.

        """

        # if what popped out is the last day, we need to update last day
        if date == self.n - 1:
            self.lastDay = self.features[-1][-1]

            # pop out the corresponding feature
            self.features.pop(self.n - 1)

        # else we can either remove the feature on the given date, but this
        # requires regenerate the new feature matrix, which might be costly.
        # Instead, we directly modify the affected entries in the feature
        # matrix.

        # The change starts from date + 1 to date + degree and
        # we should do it in a reversed order
        else:
            # all the dates that are affected and need to be corrected
            order = list(range(date + 1, min(self.n, date + self.d + 1)))

            # we reverse the processing order
            order.reverse()

            for step in order:
                # popout the deleted date
                self.features[step].pop(self.d - (step - date))

                # padding the data on the last feature to the beginning of
                # today's feature. Note for a 3-autoReg, the feature of day
                # is in a form of [day1, day2, day3]
                self.features[step].insert(0, self.features[step - 1][0])

            # pop out the corresponding feature
            self.features.pop(date)

        # popout the redundent feature and update the length
        self.n -= 1

        # check if the degree is longer than the data series
        self.checkDataLength()

    def alter(self, date, dataPoint):
        """ Alter the data of a particular date, and change the corresponding
            feature matrix.

        Args:
           date: The date to be modified.
           dataPoint: The new dataPoint to be filled in.

        """
        # if what modified is the last day, we need to update last day
        if date == self.n - 1:
            self.lastDay = dataPoint

        if dataPoint is None:
            raise NameError('The package currently do not support missing ' +
                            'for auto regression. The support has been ' +
                            'implemented, but deprecated due to efficiency ' +
                            'issue. Will support this in next version.')

        # else we can either modify the feature on the given date, but this
        # requires regenerate the new feature matrix, which might be costly.
        # Instead, we directly modify the affected entries in the feature
        # matrix.

        # The change starts from date + 1 to date + degree and
        # we should do it in a reversed order
        else:
            # all the dates that are affected and need to be corrected
            order = list(range(date + 1, min(self.n, date + self.d + 1)))

            # we reverse the processing order
            order.reverse()

            for step in order:
                # popout the deleted date
                self.features[step][self.d - (step - date)] = dataPoint

        # check if the degree is longer than the data series
        self.checkDataLength()
