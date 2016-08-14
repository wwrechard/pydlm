"""
===============================================================================

The code for the class dlm

===============================================================================

This is the main class of the dynamic linear model.
It provides the modeling, filtering, forecasting and smoothing function of a dlm.
The dlm use the @builder to construct the @baseModel based on user supplied
@components and then run @kalmanFilter to filter the result.

Example:
 # randomly generate fake data on 1000 days
> import numpy as np
> data = np.random.random((1, 1000))

 # construct the dlm of a linear trend and a 7-day seasonality
> myDlm = dlm(data) + trend(degree = 2, 0.98) + seasonality(period = 7, 0.98)

 # filter the result
> myDlm.fitForwardFilter()

 # extract the filtered result
> myDlm.getFilteredObs()

"""
# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

#from pydlm.modeler.builder import builder
from pydlm.func._dlm import _dlm
from pydlm.base.tools import duplicateList

class dlm(_dlm):
    """
    The main class of the dynamic linear model. Provide functionality for modeling,
    filtering, forecasting and smoothing.

    Members:
       See the members for @_dlm

    Methods:
       add/+: add new modeling component
       ls: list out all existing model components and names
       delete: delete one existing model components by its name
       
       getAll: get all the _result class which contains all results
       getFilteredObs: get the filtered observations
       getFilteredObsVar: get the filtered observation variance
       getFilteredState: get the filtered latent states
       getFilteredCov: get the filtered covariance matrix
       getSmoothedObs: get the smoothed observation
       getSmoothedObsVar: get the smoothed observation variance
       getSmoothedState: get the smoothed latent states
       getSmoothedCov: get the smoothed covariance matrix
       getPredictedObs: get the predicted observations (array of one-day ahead prediction)
       getPredictedObsVar: get the predicted variance (array of one-day ahead prediction)

       fitForwardFilter: fit the forward filter on the data
       fitBackwardSmoother: fit the backward smoother on the data
       fit: fit both the forward filter and the backward smoother
       predict: make prediction based on all the data

       append: append new data and features to the current 
       popout: pop out existing data of a particular days
       alter: alter the data of a specific date
       ignore: ignore the data of a specific date, treated as missing data
    """
    # define the basic members
    # initialize the result
    def __init__(self, data):
        _dlm.__init__(self, data)

#===================== modeling components =====================

    # add component
    def add(self, component):
        """
        Add new modeling component to the dlm. Currently support: trend, seasonality
        and dynamic component.

        """
        self.__add__(component)

    def __add__(self, component):    
        self.builder.__add__(component)
        self.initialized = False
        return self

    # list all components
    def ls(self):
        """
        List out all existing components

        """
        self.builder.ls()

    # delete one component
    def delete(self, name):
        """
        Delete model component by its name
        """
        self.builder.delete(name)
        self.initialized = False

#====================== result components ====================

    def getAll(self):
        """
        get all the _result class which contains all results

        """
        return self.result

    def getFilteredObs(self):
        """       
        get the filtered observations. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
        return self.result.filteredObs

    def getFilteredObsVar(self):
        """
        get the filtered observation variance. If the filtered dates are not 
        (0, self.n - 1), then a warning will prompt stating the actual filtered dates.
        """
        if self.result.filteredSteps != (0, self.n - 1):
            print 'The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1])
        return self.result.filteredObsVar

    def getSmoothedObs(self):
        """       
        get the smoothed observations. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
        return self.result.smoothedObs

    def getSmoothedObsVar(self):
        """       
        get the smoothed variance. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
        if self.result.smootehdSteps != (0, self.n - 1):
            print 'The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1])
        return self.result.smoothedObsVar

    def getPredictedObs(self):
        """       
        get the predicted observations. An array of numbers. a[k] shows the prediction on 
        that the date k given all the data up to k - 1.

        """
        return self.result.predictedObs

    def getPredictedObsVar(self):
        """       
        get the predicted variance. An array of numbers. a[k] shows the prediction on 
        that the date k given all the data up to k - 1.

        """
        return self.result.predictedObsVar

    def getFilteredState(self, name = 'all'):
        """
        get the filtered states. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
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
        """       
        get the smoothed latent states. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
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
        """
        get the filtered covariance. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
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
        """       
        get the smoothed covariance. If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        """
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
        """
        Fit forward filter on the available data. User can choose use rolling windowFront
        or not. If user choose not to use the rolling window, then the filtering
        will be based on all the previous data. If rolling window is used, then the
        filtering for a particular date will only consider previous dates that are
        within the rolling window length.
        
        Args:
            useRollingWindow: indicate whether rolling window should be used.
            windowLength: the length of the rolling window if used.

        """
        # check if the feature size matches the data size
        self._checkFeatureSize()
        
        # see if the model has been initialized
        if not self.initialized:
            self._initialize()

        print 'Starting forward filtering...'
        if not useRollingWindow:
            # we start from the last step of previous fitering
            start = self.result.filteredSteps[1] + 1
            self._forwardFilter(start = start, end = self.n - 1)
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
        self.result.filteredSteps = [0, self.n - 1]
        print 'Forward fitering completed.'
    
    def fitBackwardSmoother(self, backLength = None):
        """
        Fit backward smoothing on the data. Starting from the last observed date.
        
        Args:
            backLength: integer, indicating how many days the backward smoother
            should go, starting from the last date.

        """
        
        # see if the model has been initialized
        if not self.initialized:
            raise NameError('Backward Smoother has to be run after forward filter')

        if self.result.filteredSteps[1] != self.n - 1:
            raise NameError('Forward Fiter needs to run on full data before using backward Smoother')

        # default value for backLength        
        if backLength is None:
            backLength = self.n

        print 'Starting backward smoothing...'
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

        self.result.smootehdSteps = [self.n - backLength, self.n - 1]
        print 'Backward smoothing completed.'
            

    def fit(self):
        """
        An easy caller for fitting both the forward filter and backward smoother.

        """
        self.fitForwardFilter()
        self.fitBackwardSmoother(backLength = self.n)

#=========================== model prediction ==============================

    # The prediction function
    def predict(self, date = None, days = 1):
        
        # the default prediction date
        if date is None:
            date = self.n - 1

        # check if the data on the date has been filtered
        if date > self.result.filteredSteps[1]:
            raise NameError('Prediction can only be made right after the filtered date')
        
        return self._predict(date = date, days = days)

#======================= data appending, popping and altering ===============

    # Append new data or features to the dlm
    def append(self, data, component = 'mainData'):

        if component == 'mainData':
            # add the data to the self.data
            self.data.extend(list(data))

            # update the length
            self.n += len(data)
            self.result._appendResult(len(data))

        elif component in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[component]
            comp.features.extend(duplicateList(data))
            comp.n += len(data)

        else:
            raise NameError('Such dynamic component does not exist.')


    # pop the data of a specific date out
    def popout(self, date):

        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        # pop out the data at date
        self.data.pop(date)
        self.n -= 1

        # pop out the feature at date
        for name in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[name]
            comp.features.pop(date)
            comp.n -= 1

        # pop out the results at date
        self.result._popout(date)

        # update the filtered and the smoothed steps
        self.result.filteredSteps[1] = date - 1
        self.result.smoothedSteps[1] = date - 1

        if self.result.filteredSteps[0] > self.result.filteredSteps[1]:
            self.result.filteredSteps = [0, -1]
            self.result.smoothedSteps = [0, -1]

        elif self.result.smoothedSteps[0] > self.result.smoothedSteps[1]:
            self.result.smoothedSteps = [0, -1]

    # alter the data of a specific days
    def alter(self, date, data, component = 'mainData'):

        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        # to alter the data for the observed chain
        if component == 'mainData':
            self.data[date] = data

        # to alter the feature of a component
        elif component in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[component]
            comp.features[component] = data
            
        else:
            raise NameError('Such dynamic component does not exist.')
        
        # update the filtered and the smoothed steps
        self.result.filteredSteps[1] = date - 1
        self.result.smoothedSteps[1] = date - 1

        if self.result.filteredSteps[0] > self.result.filteredSteps[1]:
            self.result.filteredSteps = [0, -1]
            self.result.smoothedSteps = [0, -1]

        elif self.result.smoothedSteps[0] > self.result.smoothedSteps[1]:
            self.result.smoothedSteps = [0, -1]

    # ignore the data of a given date
    def ignore(self, date):

        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        self.alter(date = date, data = None, component = 'mainData')
