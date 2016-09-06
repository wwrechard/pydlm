from dynamic import dynamic

class autoReg(dynamic):

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
        features = self.createFeatureMatrix(degree = degree,
                                            data = fakeData,
                                            padding = padding)

        dynamic.__init__(features = features, discount = discount, name = name)
        self.componentType = 'autoReg'

        # the data of the last day has to be remembered, since it is not used so far and
        # will be needed when adding new data
        self.lastDay = data[-1]
        
    def createFeatureMatrix(self, degree, data):
        
        # initialize feature matrix
        features = []

        for i in range(degree, len(data)):
            features.append(data[(i - degree) : i])
                        
        return features

    def appendNewData(self, newData):
        # fetch the last entry of the feature
        previousDays = self.features[-1]
        fakeData = [num for num in previousDays] + [self.lastDay] + \
                   [num for num in newData]
        
        # delete the first day which is duplication
        fakeData.pop(0)
        newFeatures = self.createFeatureMatrix(degree = self.d, data = fakeData)

        self.features.extend(newFeatures)
        self.lastDay = newData[-1]
