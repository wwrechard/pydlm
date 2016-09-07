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
        
        # initialize feature matrix
        features = []

        for i in range(degree, len(data)):
            features.append(data[(i - degree) : i])
                        
        return features

    # the degree cannot be longer than data
    def checkDataLength(self):
        if self.d >= self.n:
            raise NameError('The degree cannot be longer than the data series')
        
    # override
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
        self.n = len(self.features)

    # override
    def popout(self, date):

        # if what popped out is the last day, we need to update last day
        if date == self.n - 1:
            self.lastDay = self.features[-1][-1]

        # else  change start from date + 1 to date + degree
        # we should reverse the order when shifting
        else:
            order = range(date + 1, date + self.d + 1)
            order.reverse()
        
            for step in order:
                self.features[step].pop(self.d - (step - date ))
                self.features[step].insert(0, self.features[step - 1][0])

        # popout the redundent feature and update the length
        self.features.pop(date)
        self.n -= 1

        # check if the degree is longer than the data series
        self.checkDataLength()
