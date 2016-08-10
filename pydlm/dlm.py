# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

#from pydlm.modeler.builder import builder
from pydlm.func._dlm import _dlm


class dlm(_dlm):

    # define the basic members
    # initialize the result
    def __init__(self, data):
        super(dlm, self).__init__(data)

#===================== modeling components =====================

    # add component
    def add(self, component):
        self.__add__(component)

    def __add__(self, component):
        self.builder.__add__(component)
        self.initialized = False
        return self

    # list all components
    def ls(self):
        self.builder.ls()

    # delete one component
    def delete(self, name):
        self.initialized = False
        self.builder.delete(name)

#====================== result components ====================

    def getAll(self):
        return self.result

    def getFilteredObs(self):
        print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + ' to ' \
                + str(self.result.filteredSteps[1])
        return self.result.filteredObs

    def getFilteredObsVar(self):
        print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + ' to ' \
                + str(self.result.filteredSteps[1])
        return self.result.filteredObsVar

    def getSmoothedObs(self):
        print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + ' to ' \
                + str(self.result.smoothedSteps[1])
        return self.result.smoothedObs

    def getSmoothedObsVar(self):
        print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + ' to ' \
                + str(self.result.smoothedSteps[1])
        return self.result.smoothedObsVar

    def getPredictedObs(self):
        return self.result.predictedObs

    def getPredictedObsVar(self):
        return self.result.predictedObsVar

    def getFilteredState(self, name = 'all'):
        print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + ' to ' \
                + str(self.result.filteredSteps[1])
        if name == 'all':
            return self.result.filteredState

        elif name in self.builder.staticComponents or \
             name in self.builder.dynamicComponents:
            indx = self.builder.componentIndex[name]
            result = [None] * self.n
            for i in range(len(result)):
                result[i] = self.result.filteredState[indx[0] : (indx[1] + 1), 0]
            return result

        else:
            raise NameError('Such component does not exist!')

    def getSmoothedState(self, name = 'all'):
        print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + ' to ' \
                + str(self.result.smoothedSteps[1])
        if name == 'all':
            return self.result.smoothedState

        elif name in self.builder.staticComponents or \
             name in self.builder.dynamicComponents:
            indx = self.builder.componentIndex[name]
            result = [None] * self.n
            for i in range(len(result)):
                result[i] = self.result.smoothedState[indx[0] : (indx[1] + 1), 0]
            return result

        else:
            raise NameError('Such component does not exist!')

    def getFilteredCov(self, name = 'all'):
        print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + ' to ' \
                + str(self.result.filteredSteps[1])
        if name == 'all':
            return self.result.filteredCov

        elif name in self.builder.staticComponents or \
             name in self.builder.dynamicComponents:
            indx = self.builder.componentIndex[name]
            result = [None] * self.n
            for i in range(len(result)):
                result[i] = self.result.filteredCov[indx[0] : (indx[1] + 1), \
                                                    indx[0] : (indx[1] + 1)]
            return result
        
        else:
            raise NameError('Such component does not exist!')

    def getSmoothedCov(self, name = 'all'):
        print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + ' to ' \
                + str(self.result.smoothedSteps[1])
        if name == 'all':
            return self.result.smoothedCov

        elif name in self.builder.staticComponents or \
             name in self.builder.dynamicComponents:
            indx = self.builder.componentIndex[name]
            result = [None] * self.n
            for i in range(len(result)):
                result[i] = self.result.smoothedCov[indx[0] : (indx[1] + 1), \
                                                    indx[0] : (indx[1] + 1)]
            return result
        
        else:
            raise NameError('Such component does not exist!')
        
#========================== model training component =======================

    def fitForwardFilter(self, useRollingWindow = False, windowLength = 3):
        # see if the model has been initialized
        if not self.initialized:
            self._initialize()
            
        start = self.result.filteredSteps[1] + 1
        if not useRollingWindow:
            # we start from the last step of previous fitering
            self._forwardFilter(start = start, end = self.n - 1)
        else:
            # for dates within (start, windowLength - 1) we ran the usual ff
            self._forwardFilter(start = start, end = windowLength - 1)

            # for the remaining date, we use a rolling window
            for day in range(max(start, windowLength), self.n):
                self._forwardFilter(start = max(0, day - windowLength + 1), \
                                       save = day, ForgetPrevious = True)

    
    def fitBackwardSmoother(self, backLength = 3):
        # see if the model has been initialized
        if not self.initialized:
            raise NameError('Backward Smoother has to be run after forward filter')

        if self.result.filteredSteps[1] != self.n - 1:
            raise NameError('Forward Fiter needs to run on full data before using backward Smoother')

        if self.result.smoothedSteps[1] == self.n - 1:
            return None
        else:
            self._backwardSmoother(start = self.n - 1, days = backLength)

    def fit(self):
        self.fitForwardFilter()
        self.fitBackwardSmoother(backLength = self.n)

#=========================== model prediction ==============================

#    def predict(self, date = None, days = 1):
