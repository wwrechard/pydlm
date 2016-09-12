# -*- coding: utf-8 -*-
"""
===============================================================================

The code for the class dlm

===============================================================================

This is the main class of the Bayeisan dynamic linear model.
It provides the modeling, filtering, forecasting and smoothing function of a dlm.
The dlm use the @builder to construct the @baseModel based on user supplied
@components and then run @kalmanFilter to filter the result.

Example:
>>> # randomly generate fake data on 1000 days
>>> import numpy as np
>>> data = np.random.random((1, 1000))

>>> # construct the dlm of a linear trend and a 7-day seasonality
>>> myDlm = dlm(data) + trend(degree = 2, 0.98) + seasonality(period = 7, 0.98)

>>> # filter the result
>>> myDlm.fitForwardFilter()

>>> # extract the filtered result
>>> myDlm.getFilteredObs()

"""
# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

#from pydlm.modeler.builder import builder
from pydlm.func._dlm import _dlm
#from pydlm.base.tools import duplicateList
from pydlm.base.tools import getInterval

class dlm(_dlm):
    """ The main class of the dynamic linear model. 


    This is the main class of the Bayeisan dynamic linear model.
    It provides the modeling, filtering, forecasting and smoothing function of a dlm.
    The dlm use the @builder to construct the @baseModel based on user supplied
    @components and then run @kalmanFilter to filter the result.

    Example:
        >>> # randomly generate fake data on 1000 days
        >>> import numpy as np
        >>> data = np.random.random((1, 1000))    
        >>> # construct the dlm of a linear trend and a 7-day seasonality
        >>> myDlm = dlm(data) + trend(degree = 2, 0.98) + seasonality(period = 7, 0.98)    
        >>> # filter the result
        >>> myDlm.fitForwardFilter()
        >>> # extract the filtered result
        >>> myDlm.getFilteredObs()
    
    Attributes:
       data: a list of doubles of the raw time series data. 
             It could be either the python's built-in list of doubles or numpy 1d array.

    """
    # define the basic members
    # initialize the result
    def __init__(self, data):
        _dlm.__init__(self, data)

#===================== modeling components =====================

    # add component
    def add(self, component):
        """ Add new modeling component to the dlm. 

        Currently support: trend, seasonality, autoregression and dynamic component.

        Args:
            component: the modeling component, could be either one of the following:\n
                       trend, seasonality, dynamic, autoReg. 

        Returns:
            A dlm object with added component.

        """
        self.__add__(component)

    def __add__(self, component):    
        self.builder.__add__(component)
        self.initialized = False
        return self

    # list all components
    def ls(self):
        """ List out all existing components

        """
        self.builder.ls()

    # delete one component
    def delete(self, name):
        """ Delete model component by its name
        
        Args:
            name: the name of the component.

        """
        self.builder.delete(name)
        self.initialized = False

#====================== result components ====================

    def getAll(self):
        """ get all the _result class which contains all results

        Returns:
            The @result object containing all computed results.

        """
        return self.result

    def getFilteredObs(self):
        """ get the filtered observations. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Returns:
            A list of the filtered time series observations

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        # get out of the matrix form
        return self._1DmatrixToArray(self.result.filteredObs[start:end])

    def getFilteredObsVar(self):
        """ get the filtered observation variance. 

        If the filtered dates are not (0, self.n - 1), then a warning will prompt stating
        the actual filtered dates.

        Returns:
            A list of the filtered time series variances.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        return self._1DmatrixToArray(self.result.filteredObsVar[start:end])

    def getFilteredInterval(self, p):
        """ get the filtered confidence interval. 

        If the filtered dates are not 
        (0, self.n - 1), then a warning will prompt stating the actual filtered dates.

        Args:
            p: The confidence level.

        Returns:
            A tuple with the first element being a list of upper bounds 
            and the second being a list of the lower bounds.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        upper, lower =  getInterval(self.result.filteredObs[start : end], \
                                    self.result.filteredObsVar[start : end], p)
        return (self._1DmatrixToArray(upper), self._1DmatrixToArray(lower))

    def getSmoothedObs(self):
        """ get the smoothed observations. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Returns:
            A list of the smoothed time series observations.

        """
        if self.result.smoothedSteps != [0, self.n - 1]:
            print('The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1]))
        start = self.result.smoothedSteps[0]
        end = self.result.smoothedSteps[1] + 1
        return self._1DmatrixToArray(self.result.smoothedObs[start:end])

    def getSmoothedObsVar(self):
        """ get the smoothed variance. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Returns:
            A list of the smoothed time series variances.

        """
        if self.result.smoothedSteps != [0, self.n - 1]:
            print('The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1]))
        start = self.result.smoothedSteps[0]
        end = self.result.smoothedSteps[1] + 1
        return self._1DmatrixToArray(self.result.smoothedObsVar[start:end])

    def getSmoothedInterval(self, p):
        """ get the smoothed confidence interval. 

        If the filtered dates are not 
        (0, self.n - 1), then a warning will prompt stating the actual smoothed dates.

        Args:
            p: The confidence level.

        Returns:
            A tuple with the first element being a list of upper bounds 
            and the second being a list of the lower bounds.

        """
        if self.result.smoothedSteps != [0, self.n - 1]:
            print('The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1]))
        start = self.result.smoothedSteps[0]
        end = self.result.smoothedSteps[1] + 1
        upper, lower = getInterval(self.result.smoothedObs[start : end], \
                                   self.result.smoothedObsVar[start : end], p)
        return (self._1DmatrixToArray(upper), self._1DmatrixToArray(lower))
    
    def getPredictedObs(self):
        """ get the predicted observations. 

        An array of numbers. 
        a[k] shows the prediction on that the date k given all the data up to k - 1.

        Returns:
            A list of one-day ahead predictions.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The predicted dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        return self._1DmatrixToArray(self.result.predictedObs[start : end])

    def getPredictedObsVar(self):
        """ get the predicted variance. 

        An array of numbers. a[k] shows the prediction on 
        that the date k given all the data up to k - 1.

        Returns:
            A list of one-day ahead prediction variances.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The predicted dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        return self._1DmatrixToArray(self.result.predictedObsVar[start : end])

    def getPredictedInterval(self, p):
        """ get the predicted confidence interval. 

        If the predicted dates are not 
        (0, self.n - 1), then a warning will prompt stating the actual predicted dates.

        Args:
            p: The confidence level.

        Returns:
            A tuple with the first element being a list of upper bounds 
            and the second being a list of the lower bounds.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The predicted dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
        start = self.result.filteredSteps[0]
        end = self.result.filteredSteps[1] + 1
        upper, lower = getInterval(self.result.predictedObs[start : end], \
                                   self.result.predictedObsVar[start : end], p)
        return (self._1DmatrixToArray(upper), self._1DmatrixToArray(lower))

    def getFilteredState(self, name = 'all'):
        """ get the filtered states. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            name: the component name for which to get the filtered results. 
                  Default to 'all'.

        Returns:
            A list of numpy matrices, standing for the filtered latent states.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
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
        """ get the smoothed latent states. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            name: the component for which to get the smoothed results.
                  Default to 'all'.

        Returns:
            A list of numpy matrices, standing for the smoothed latent states.
        
        """
        if self.result.smootehdSteps != [0, self.n - 1]:
            print('The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1]))
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
        """ get the filtered error covariance. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            name: the componnet name for which to ge the filtered state covariance.
                  Default to be 'all'.

        Returns:
            A list of numpy matrices, standing for the filtered latent covariance.

        """
        if self.result.filteredSteps != [0, self.n - 1]:
            print('The fitlered dates are from ' + str(self.result.filteredSteps[0]) + \
                ' to ' + str(self.result.filteredSteps[1]))
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
        """ get the smoothed covariance. 

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            name: the componnet name for which to ge the smoothed state covariance.
                  Default to be 'all'.

        Returns:
            A list of numpy matrices, standing for the smoothed latent covariance.

        """
        if self.result.smootehdSteps != [0, self.n - 1]:
            print('The smoothed dates are from ' + str(self.result.smoothedSteps[0]) + \
                ' to ' + str(self.result.smoothedSteps[1]))
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
        """ Fit forward filter on the available data. 

        User can choose use rolling windowFront
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

        print('Starting forward filtering...')
        if not useRollingWindow:
            # we start from the last step of previous fitering
            if self.result.filteredType == 'non-rolling':
                start = self.result.filteredSteps[1] + 1
            else:
                start = 0

            # determine whether renew should be used
            self._forwardFilter(start = start, \
                                end = self.n - 1, \
                                renew = self.options.stable)
            self.result.filteredType = 'non-rolling'
        else:
            if self.result.filteredType == 'rolling':
                windowFront = self.result.filteredSteps[1] + 1
            else:
                windowFront = 0
            self.result.filteredType = 'rolling'
            # if end is still within (0, windowLength - 1), we should run the
            # usual ff from
            if windowFront < windowLength:
                self._forwardFilter(start = self.result.filteredSteps[1] + 1, \
                                    end = min(windowLength - 1, self.n - 1))

            # for the remaining date, we use a rolling window
            for day in range(max(windowFront, windowLength), self.n):
                self._forwardFilter(start = day - windowLength + 1, \
                                    end = day, \
                                    save = day, \
                                    ForgetPrevious = True)
                
        self.result.filteredSteps = [0, self.n - 1]        
        self.turnOn('filtered plot')
        self.turnOn('predict plot')
        print('Forward fitering completed.')
    
    def fitBackwardSmoother(self, backLength = None):
        """ Fit backward smoothing on the data. Starting from the last observed date.
        
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

        print('Starting backward smoothing...')
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

        self.result.smoothedSteps = [self.n - backLength, self.n - 1]
        self.turnOn('smoothed plot')
        print('Backward smoothing completed.')
            

    def fit(self):
        """ An easy caller for fitting both the forward filter and backward smoother.

        """
        self.fitForwardFilter()
        self.fitBackwardSmoother()

#=========================== model prediction ==============================

    # The prediction function
    def predict(self, date = None, days = 1):
        """ Predict based on the current data for a specific future days

        The predict result is based on all the data before date and predict the
        observation at date + days.

        Args:
            date: the index when the prediction starts. Default to the last day.
            days: number of days ahead to predict

        Returns:
            A tuple. (Predicted observation, variance of the predicted observation)

        """
        # the default prediction date
        if date is None:
            date = self.n - 1

        # check if the data on the date has been filtered
        if date > self.result.filteredSteps[1]:
            raise NameError('Prediction can only be made right after the filtered date')
        
        return self._predict(date = date, days = days)

#======================= data appending, popping and altering ===============

    # Append new data or features to the dlm
    def append(self, data, component = 'main'):
        """ Append the new data to the main data or the components (new feature data)

        Args:
            data: the new data
            component: the name of which the new data to be added to.\n
                       'main': the main time series data\n
                       other omponent name: add new feature data to other component.

        """
        if component == 'main':
            # add the data to the self.data
            self.data.extend(list(data))

            # update the length
            self.n += len(data)
            self.result._appendResult(len(data))

        elif component in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[component]
            comp.appendNewData(data)

        else:
            raise NameError('Such dynamic component does not exist.')


    # pop the data of a specific date out
    def popout(self, date):
        """ Pop out the data for a given date
        
        Args:
            date: the index indicates which date to be popped out.

        """
        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        # pop out the data at date
        self.data.pop(date)
        self.n -= 1

        # pop out the feature at date
        for name in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[name]
            comp.popout(date)

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
    def alter(self, date, data, component = 'main'):
        """ To alter the data for a specific date and a specific component.

        Args:
            date: the date of the altering data
            data: the new data
            component: the component for which the new data need to be supplied to.\n
                       'main': the main time series data\n
                       other component name: other component feature data

        """
        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        # to alter the data for the observed chain
        if component == 'main':
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
        """ Ignore the data for a specific day. treat it as missing data

        Args:
            date: the date to ignore.

        """
        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' + str(self.n - 1))

        self.alter(date = date, data = None, component = 'main')

#============================= plot component ====================================

    def turnOn(self, switch):
        """ "turn on" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch on. \n
                    'filtered plot', 'filter' to plot filtered results\n
                    'smoothed plot', 'smooth' to plot smoothed results\n
                    'predict plot', 'predict', to plot one-step ahead results\n
                    'confidence interval', 'confierence', 'CI' to plot CI's\n
                    'data points', 'data', 'data point' to plot original data\n
                    'multiple', 'separate' to plot results in separate figures\n
                    'fitted dots', 'fitted' to plot fitted results with dots
        """
        if switch in set(['filtered plot', 'filter', 'filtered results', 'filtering']):
            self.options.plotFilteredData = True
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results', 'smoothing']):
            self.options.plotSmoothedData = True
        elif switch in set(['predict plot', 'predict', 'predicted results', 'prediction']):
            self.options.plotPredictedData = True
        elif switch in set(['confidence interval', 'confidence', 'interval', 'CI', 'ci']):
            self.options.showConfidenceInterval = True
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = True
        elif switch in set(['multiple', 'multiple plots', 'separate plots', 'separate']):
            self.options.separatePlot = True
        elif switch in set(['fitted dots', 'fitted results', 'fitted data', 'fitted']):
            self.options.showFittedPoint = True
        else:
            raise NameError('no such options')

    def turnOff(self, switch):
        """ "turn off" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch off. \n
                    'filtered plot', 'filter' to not plot filtered results\n
                    'smoothed plot', 'smooth' to not plot smoothed results\n
                    'predict plot', 'predict', to not plot one-step ahead results\n
                    'confidence interval', 'confierence', 'CI' to not plot CI's\n
                    'data points', 'data', 'data point' to not plot original data\n
                    'multiple', 'separate' to not plot results in separate figures\n
                    'fitted dots', 'fitted' to not plot fitted results with dots

        """
        if switch in set(['filtered plot', 'filter', 'filtered results', 'filtering']):
            self.options.plotFilteredData = False
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results', 'smoothing']):
            self.options.plotSmoothedData = False
        elif switch in set(['predict plot', 'predict', 'predicted results', 'prediction']):
            self.options.plotPredictedData = False
        elif switch in set(['confidence interval', 'confidence', 'interval', 'CI', 'ci']):
            self.options.showConfidenceInterval = False
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = False
        elif switch in set(['multiple', 'multiple plots', 'separate plots', 'separate']):
            self.options.separatePlot = False
        elif switch in set(['fitted dots', 'fitted results', 'fitted data', 'fitted']):
            self.options.showFittedPoint = False
        else:
            raise NameError('no such options')

    def setColor(self, switch, color):
        """ "set" Operation for the dlm plotting colors

        Args:
            switch: key word. Controls over filtered/smoothed/predicted results, 
            color: the color for the corresponding keyword.
        """
        if switch in set(['filtered plot', 'filter', 'filtered results', 'filtering']):
            self.options.filteredColor = color
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results', 'smoothing']):
            self.options.smoothedColor = color
        elif switch in set(['predict plot', 'predict', 'predicted results', 'prediction']):
            self.options.predictedColor = color
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.dataColor = color
        else:
            raise NameError('no such options')

    def setConfidence(self, p = 0.95):
        """ Set the confidence interval for the plot

        """
        assert p >= 0 and p <= 1
        self.options.confidence = p
        
    def resetPlotOptions(self):
        """ Reset the plotting option for the dlm class

        """
        self.options.plotOriginalData = True
        self.options.plotFilteredData = True
        self.options.plotSmoothedData = True
        self.options.plotPredictedData = True
        self.options.showDataPoint = True
        self.options.showFittedPoint = False
        self.options.showConfidenceInterval = True
        self.options.dataColor = 'black'
        self.options.filteredColor = 'blue'
        self.options.predictedColor = 'green'
        self.options.smoothedColor = 'red'
        self.options.separatePlot = True
        self.options.confidence = 0.95
        
    # plot the result according to the options
    def plot(self):
        """ The main plot function. The dlmPlot and the matplotlib will only be loaded
        when necessary.

        """

        # load the library only when needed
        import pydlm.plot.dlmPlot as dlmPlot
        
        if self.time is None:
            time = range(len(self.data))

        # initialize the figure
        dlmPlot.plotInitialize()

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and fitted before ploting.')
            
        if self.result.filteredSteps[1] == -1:
            self.options.plotFilteredData = False
            self.options.plotPredictedData = False

        if self.result.smoothedSteps[1] == -1:
            self.options.plotSmoothedData = False
            
        # if we just need one plot
        if self.options.separatePlot is not True:
            dlmPlot.plotInOneFigure(time = time, \
                                    data = self.data, \
                                    result = self.result, \
                                    options = self.options)
        else:
            dlmPlot.plotInMultipleFigure(time = time, \
                                         data = self.data, \
                                         result = self.result, \
                                         options = self.options)

        dlmPlot.plotout()

#================================ control options ==================================
    def showOptions(self):
        """ Print out all the option values

        """
        allItems = vars(self.options)
        for item in allItems:
            print(item + ': ' + str(allItems[item]))

    def stableMode(self, use = True):
        """ Turn on the stable mode, i.e., using the renewal strategy.

            Indicate whether the renew strategy should be used to add numerical
            stability. When the filter goes over certain steps,
            the information contribution of the previous data has decayed to minimum.
            In the stable mode, We then ignore those days and refit the time series 
            starting from current - renewTerm, where renewTerm is computed according
            to the discount. Thus, the effective sample size of the dlm is twice
            renewTerm. When discount = 1, there will be no renewTerm, since all the
            information will be passed along.
        """
        if self.options.stable != use:
            self.initialized = False
        self.options.stable = use
