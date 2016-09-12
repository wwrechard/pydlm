"""
===========================================================================

The code for autoregressive components

===========================================================================

This code implements the autoregressive component as a sub-class of dynamic.
Different from the dynamic component, the features in the autoReg is generated
from the data, and updated according to the data. All other features are similar
to @dynamic.

"""

from .dynamic import dynamic

class autoReg(dynamic):
    """ The autoReg class alows user to add an autoregressive component to the dlm. 
    This code implements the autoregressive component as a sub-class of dynamic.
    Different from the dynamic component, the features in the autoReg is generated
    from the data, and updated according to the data. All other features are similar
    to @dynamic.

    Attributes:
        degree: the degree of autoregressive, i.e., how many days to look back
        data: the time series data used for constructing the autoregressive features
        discount factor: the discounting factor
        name: the name of the component
        padding: either 0 or None. The number to be padded for the first degree days,
                 as no previous data is observed to form the feature matrix

    """

    def __init__(self,
                 degree = 2,
                 data = None,
                 discount = 0.99,
                 name = 'ar2',
                 padding = 0):
        
        if data is None:
            raise NameError('data must be provided to construct autoregressive component')
        
        # create fake data to incorporate paddings in the beginning
        fakeData = [padding] * degree + [num for num in data]

        # create features
        features = self.createFeatureMatrix(degree = degree, data = fakeData)

        dynamic.__init__(self,
                         features = features,
                         discount = discount,
                         name = name)
        self.checkDataLength()
        
        # the data of the last day has to be remembered, since it is not used so far and
        # will be needed when adding new data
        self.lastDay = data[-1]
        
    def createFeatureMatrix(self, degree, data):
        """ Create the feature matrix based on the supplied data and the degree.

        Args:
            degree: the auto-regressive dependency length.
            data: the raw time series data of the model.
        """
        
        # initialize feature matrix
        features = []

        for i in range(degree, len(data)):
            features.append(data[(i - degree) : i])
                        
        return features

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
        # fetch the last entry of the feature
        previousDays = self.features[-1]
        fakeData = [num for num in previousDays] + [self.lastDay] + \
                   [num for num in newData]
        
        # delete the first day which is duplication
        fakeData.pop(0)
        newFeatures = self.createFeatureMatrix(degree = self.d, data = fakeData)

        self.features.extend(newFeatures)
        self.lastDay = newData[-1]
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

        # else  change start from date + 1 to date + degree
        # we should reverse the order when shifting
        else:
            order = list(range(date + 1, date + self.d + 1))
            order.reverse()
        
            for step in order:
                self.features[step].pop(self.d - (step - date ))
                self.features[step].insert(0, self.features[step - 1][0])

        # popout the redundent feature and update the length
        self.features.pop(date)
        self.n -= 1

        # check if the degree is longer than the data series
        self.checkDataLength()
