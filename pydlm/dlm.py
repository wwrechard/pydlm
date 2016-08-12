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
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
        return self.result.filteredObs

    def getFilteredObsVar(self):
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
        return self.result.filteredObsVar

    def getSmoothedObs(self):
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
        return self.result.smoothedObs

    def getSmoothedObsVar(self):
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
        return self.result.smoothedObsVar

    def getPredictedObs(self):
        return self.result.predictedObs

    def getPredictedObsVar(self):
        return self.result.predictedObsVar

    def getFilteredState(self, name = 'all'):
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
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
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
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
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
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
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
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
            
        if not useRollingWindow:
            # we start from the last step of previous fitering
            start = self.result.filteredSteps[1] + 1
            self._forwardFilter(start = start, end = self.n - 1)
            self.result.filteredSteps = (0, self.n - 1)
        else:
            windowFront = self.result.filteredSteps[1] + 1
            # if end is still within (0, windowLength - 1), we should run the
            # usual ff from
            if windowFront < windowLength:
                self._forwardFilter(start = self.result.filteredSteps[1] + 1, \
                                    end = min(windowLength - 1, self.n - 1))

            else:
            # for the remaining date, we use a rolling window
                for day in range(max(windowFront, windowLength), self.n):
                    self._forwardFilter(start = day - windowLength + 1, \
                                        end = day, \
                                        save = day, \
                                        ForgetPrevious = True)
            self.result.filteredSteps = (0, self.n - 1)

    
    def fitBackwardSmoother(self, backLength = None):
        # see if the model has been initialized
        if not self.initialized:
            raise NameError('Backward Smoother has to be run after forward filter')

        if self.result.filteredSteps[1] != self.n - 1:
            raise NameError('Forward Fiter needs to run on full data before using backward Smoother')

        # default value for backLength        
        if backLength is None:
            backLength = self.n
            
        # if the smoothed dates has already been done, we do nothing
        if self.result.smoothedSteps[1] == self.n - 1 and \
           self.result.smoothedSteps[0] <= self.n - 1 - backLength + 1:
            return None

        # if the smoothed dates start from n - 1, we just need to continue
        elif self.result.smoothedSteps[1] == self.n - 1:
            self._backwardSmoother(start = self.result.smoothedSteps[0] - 1, \
                                   days = backLength)
            
        # if the smoothed dates are even earlier, we need to start from the beginning
        elif self.result.smoothedSteps[1] < self.n - 1:
            self._backwardSmoother(start = self.n - 1, days = backLength)

        self.result.smootehdSteps = (self.n - backLength, self.n - 1)
            

    def fit(self):
        self.fitForwardFilter()
        self.fitBackwardSmoother(backLength = self.n)

#=========================== model prediction ==============================

    def predict(self, date = None, days = 1):
        if date is None:
            date = self.n - 1

        
