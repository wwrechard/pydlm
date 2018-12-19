# -*- coding: utf-8 -*-
"""
===============================================================================

The code for the class dlm

===============================================================================

This is the main class of the Bayeisan dynamic linear model.
It provides the modeling, filtering, forecasting and smoothing function
 of a dlm. The dlm use the @builder to construct the @baseModel based
 on user supplied @components and then run @kalmanFilter to filter the result.

Example:
>>> # randomly generate fake data on 1000 days
>>> import numpy as np
>>> data = np.random.random((1, 1000))

>>> # construct the dlm of a linear trend and a 7-day seasonality
>>> from pydlm import dlm, trend, seasonality
>>> myDlm = dlm(data) + trend(degree = 2, 0.98) + seasonality(period = 7, 0.98)

>>> # filter the result
>>> myDlm.fitForwardFilter()

>>> # extract the filtered result
>>> myDlm.getFilteredObs()

"""
# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

from copy import deepcopy
from numpy import matrix
from pydlm.base.tools import getInterval
from pydlm.func._dlmPredict import _dlmPredict
from pydlm.func._dlmGet import _dlmGet
from pydlm.func._dlmTune import _dlmTune
from pydlm.tuner.dlmTuner import modelTuner

class dlm(_dlmPredict, _dlmGet, _dlmTune):
    """ The main class of the dynamic linear model.


    This is the main class of the Bayeisan dynamic linear model.
    It provides the modeling, filtering, forecasting and smoothing
    function of a dlm.
    The dlm use the @builder to construct the @baseModel based on user supplied
    @components and then run @kalmanFilter to filter the result.

    Example 1:
        >>> # randomly generate fake data on 1000 days
        >>> import numpy as np
        >>> data = np.random.random((1, 1000))
        >>> # construct the dlm of a linear trend and a 7-day seasonality
        >>> myDlm = dlm(data) + trend(degree = 2, 0.98) + seasonality(period = 7, 0.98)
        >>> # filter the result
        >>> myDlm.fitForwardFilter()
        >>> # extract the filtered result
        >>> myDlm.getFilteredObs()

    Example 2 (fit a linear regression):
        >>> from pydlm import dynamic
        >>> data = np.random.random((1, 100))
        >>> mydlm = dlm(data) + trend(degree=1, 0.98, name='a') +
                        dynamic(features=[[i] for i in range(100)], 1, name='b')
        >>> mydlm.fit()
        >>> coef_a = mydlm.getLatentState('a')
        >>> coef_b = mydlm.getLatentState('b')

    Attributes:
       data: a list of doubles of the raw time series data.
             It could be either the python's built-in list of
             doubles or numpy 1d array.

    """
    # define the basic members
    # initialize the result
    def __init__(self, data, **options):
        super(dlm, self).__init__(data, **options)

        # indicate whether the plot modules has been loaded.
        # We add this flag, since we only import plot module
        # when they are called to avoid any error due to
        # plot that blocks using this package. This flag can
        # help without doing import-check (expensive) every time
        # when plot function is called.
        self.plotLibLoaded = False

        # This model is used for prediction. Prediction functions
        # will change the model status to forecast at a particular
        # date. Using a copied model will help the main model from
        # being changed and behaving abnormally.
        self._predictModel = None

    def exportModel(self):
        """ Export the dlm builder. Currently the method only support dlm without
        dynamic components.

        """
        if length(self.builder.dynamicComponents) > 0:
            raise ValueError('Cannot export dlm builder with dynamic components.')

        if not self.initialized:
            raise ValueError('Cannot export dlm before the model was initilized.')

        return deepcopy(self.builder)

    def buildFromModel(self, model):
        """ Construct the dlm with exported model from other DLM with status.

        Args:
            model: The exported model from other dlm. Must be the return from
                   dlm.exportModel()

        """
        self._initializeFromBuilder(exported_builder=model)

# ===================== modeling components =====================

    # add component
    def add(self, component):
        """ Add new modeling component to the dlm.

        Currently support: trend, seasonality, autoregression
        and dynamic component.

        Args:
            component: the modeling component, could be either one
                       of the following:\n
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

# ========================== model training component =======================

    def fitForwardFilter(self, useRollingWindow=False, windowLength=3):
        """ Fit forward filter on the available data.

        User can choose use rolling windowFront
        or not. If user choose not to use the rolling window,
        then the filtering will be based on all the previous data.
        If rolling window is used, then the filtering for a particular
        date will only consider previous dates that are
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

        if self._printInfo:
            print('Starting forward filtering...')
        if not useRollingWindow:
            # we start from the last step of previous filtering
            if self.result.filteredType == 'non-rolling':
                start = self.result.filteredSteps[1] + 1
            else:
                start = 0
                # because we refit the forward filter, we need to reset the
                # backward smoother as well.
                self.result.smoothedSteps = [0, -1]

            # determine whether renew should be used
            self._forwardFilter(start=start,
                                end=self.n - 1,
                                renew=self.options.stable)
            self.result.filteredType = 'non-rolling'
        else:
            if self.result.filteredType == 'rolling':
                windowFront = self.result.filteredSteps[1] + 1
            else:
                windowFront = 0
                # because we refit the forward filter, we need to reset the
                # backward smoother as well.
                self.result.smoothedSteps = [0, -1]

            self.result.filteredType = 'rolling'
            # if end is still within (0, windowLength - 1), we should run the
            # usual ff from
            if windowFront < windowLength:
                self._forwardFilter(start=self.result.filteredSteps[1] + 1,
                                    end=min(windowLength - 1, self.n - 1))

            # for the remaining date, we use a rolling window
            for today in range(max(windowFront, windowLength), self.n):
                self._forwardFilter(start=today - windowLength + 1,
                                    end=today,
                                    save=today,
                                    ForgetPrevious=True)

        self.result.filteredSteps = [0, self.n - 1]
        self.turnOn('filtered plot')
        self.turnOn('predict plot')

        if self._printInfo:
            print('Forward filtering completed.')

    def fitBackwardSmoother(self, backLength=None):
        """ Fit backward smoothing on the data. Starting from the last observed date.

        Args:
            backLength: integer, indicating how many days the backward smoother
            should go, starting from the last date.

        """

        # see if the model has been initialized
        if not self.initialized:
            raise NameError('Backward Smoother has to be run after' +
                            ' forward filter')

        if self.result.filteredSteps[1] != self.n - 1:
            raise NameError('Forward Fiter needs to run on full data before' +
                            'using backward Smoother')

        # default value for backLength
        if backLength is None:
            backLength = self.n

        if self._printInfo:
            print('Starting backward smoothing...')
        # if the smoothed dates has already been done, we do nothing
        if self.result.smoothedSteps[1] == self.n - 1 and \
           self.result.smoothedSteps[0] <= self.n - 1 - backLength + 1:
            return None

        # if the smoothed dates start from n - 1, we just need to continue
        elif self.result.smoothedSteps[1] == self.n - 1:
            self._backwardSmoother(start=self.result.smoothedSteps[0] - 1,
                                   days=backLength)

        # if the smoothed dates are even earlier,
        # we need to start from the beginning
        elif self.result.smoothedSteps[1] < self.n - 1:
            self._backwardSmoother(start=self.n - 1, days=backLength)

        self.result.smoothedSteps = [self.n - backLength, self.n - 1]
        self.turnOn('smoothed plot')

        if self._printInfo:
            print('Backward smoothing completed.')

    def fit(self):
        """ An easy caller for fitting both the forward filter and backward smoother.

        """
        self.fitForwardFilter()
        self.fitBackwardSmoother()

# =========================== model prediction ==============================

    # One day ahead prediction function
    def predict(self, date=None, featureDict=None):
        """ One day ahead predict based on the current data.

        The predict result is based on all the data before date and predict the
        observation at date + days. 

        The prediction could be on the last day and into the future or in 
        the middle of the time series and ignore the rest. For predicting into
        the future, the new features must be supplied to featureDict. For 
        prediction in the middle, the user can still supply the features which
        will be used priorily. The old features will be used if featureDict is
        None.

        Args:
            date: the index when the prediction based on. Default to the
                  last day.
            featureDict: the feature set for the dynamic Components, in a form
                  of {"component_name": feature}. If the featureDict is not
                  supplied, then the algo reuse those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.

        Returns:
            A tuple. (Predicted observation, variance of the predicted
            observation)

        """
        # the default prediction date
        if date is None:
            date = self.n - 1

        # check if the data on the date has been filtered
        if date > self.result.filteredSteps[1]:
            raise NameError('Prediction can only be made right' +
                            ' after the filtered date')

        # Clear the existing predictModel before the deepcopy to avoid recurrent
        # recurrent copy which could explode the memory and complexity.
        self._predictModel = None
        self._predictModel = deepcopy(self)
        return self._predictModel._oneDayAheadPredict(date=date,
                                                      featureDict=featureDict)
        
    def continuePredict(self, featureDict=None):
        """ Continue prediction after the one-day ahead predict.

        If users want to have a multiple day prediction, they can opt to use
        continuePredict after predict with new features contained in
        featureDict. For example,

        >>> # predict 3 days after the last day
        >>> myDLM.predict(featureDict=featureDict_day1)
        >>> myDLM.continuePredict(featureDict=featureDict_day2)
        >>> myDLM.continuePredict(featureDict=featureDict_day3)

        The featureDict acts the same way as in predict().

        Args:
            featureDict: the feature set for the dynamic components, stored
                         in a for of {"component name": vector}. If the set
                         was not supplied, then the algo will re-use the old
                         feature. For days beyond the data, the featureDict
                         for every dynamic component must be provided.

        Returns:
            A tupe. (predicted observation, variance)
        """
        if self._predictModel is None:
            raise NameError('continuePredict has to come after predict.')

        return self._predictModel._continuePredict(featureDict=featureDict)

    # N day ahead prediction
    def predictN(self, N=1, date=None, featureDict=None):
        """ N day ahead prediction based on the current data.

        This function is a convenient wrapper of predict() and
        continuePredict(). If the prediction is into the future, i.e, > n, 
        the featureDict has to contain all feature vectors for multiple days
        for each dynamic component. For example, assume myDLM has a component
        named 'spy' which posseses two dimensions,

        >>> featureDict_3day = {'spy': [[1, 2],[2, 3],[3, 4]]}
        >>> myDLM.predictN(N=3, featureDict=featureDict_3day)

        Args:
            N:    The length of days to predict.
            date: The index when the prediction based on. Default to the
                  last day.
            FeatureDict: The feature set for the dynamic Components, in a form
                  of {"component_name": feature}, where the feature must have
                  N elements of feature vectors. If the featureDict is not
                  supplied, then the algo reuse those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.

        Returns:
            A tuple of two lists. (Predicted observation, variance of the predicted
            observation)

        """
        if N < 1:
            raise NameError('N has to be greater or equal to 1')
        # Take care if features are numpy matrix
        if featureDict is not None:
            for name in featureDict:
                if isinstance(featureDict[name], matrix):
                    featureDict[name] = featureDict[name].tolist()
        predictedObs = []
        predictedVar = []

        # call predict for the first day
        getSingleDayFeature = lambda f, i: ({k: v[i] for k, v in f.items()}
                                            if f is not None else None)
        # Construct the single day featureDict
        featureDictOneDay = getSingleDayFeature(featureDict, 0)
        (obs, var) = self.predict(date=date, featureDict=featureDictOneDay)
        predictedObs.append(obs)
        predictedVar.append(var)

        # Continue predicting the remaining days
        for i in range(1, N):
            featureDictOneDay = getSingleDayFeature(featureDict, i)
            (obs, var) = self.continuePredict(featureDict=featureDictOneDay)
            predictedObs.append(obs)
            predictedVar.append(var)
        return (self._1DmatrixToArray(predictedObs),
                self._1DmatrixToArray(predictedVar))

# =========================== result components =============================

    def getAll(self):
        """ get all the _result class which contains all results

        Returns:
            The @result object containing all computed results.

        """
        return deepcopy(self.result)

    def getMean(self, filterType='forwardFilter', name='main'):
        """ get mean for data or component.

        If the working dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of mean to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get mean. When name = 'main', then it
                  returns the filtered mean for the time series. When
                  name = some component's name, then it returns the filtered
                  mean for that component. Default to 'main'.

        Returns:
            A list of the time series observations based on the choice

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1 # To get the result for the last date.
        # get the mean for the fitlered data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                return self._1DmatrixToArray(
                    self.result.filteredObs[start:end])
            elif filterType == 'backwardSmoother':
                return self._1DmatrixToArray(
                    self.result.smoothedObs[start:end])
            elif filterType == 'predict':
                return self._1DmatrixToArray(
                    self.result.predictedObs[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the mean for the component
        self._checkComponent(name)
        return self._getComponentMean(name=name,
                                      filterType=filterType,
                                      start=start, end=(end - 1))

    def getVar(self, filterType='forwardFilter', name='main'):
        """ get the variance for data or component.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating
        the actual filtered dates.

        Args:
            filterType: the type of variance to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get variance. When name = 'main', then it
                  returns the filtered variance for the time series. When
                  name = some component's name, then it returns the filtered
                  variance for that component. Default to 'main'.

        Returns:
            A list of the filtered variances based on the choice.

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # get the variance for the time series data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                return self._1DmatrixToArray(
                    self.result.filteredObsVar[start:end])
            elif filterType == 'backwardSmoother':
                return self._1DmatrixToArray(
                    self.result.smoothedObsVar[start:end])
            elif filterType == 'predict':
                return self._1DmatrixToArray(
                    self.result.predictedObsVar[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the variance for the component
        self._checkComponent(name)
        return self._getComponentVar(name=name, filterType=filterType,
                                     start=start, end=(end - 1))

    def getResidual(self, filterType='forwardFilter'):
        """ get the residuals for data after filtering or smoothing.

        If the working dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of residuals to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.

        Returns:
            A list of residuals based on the choice

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1 # To get the result for the last date.
        # get the mean for the fitlered data
        # get out of the matrix form
        if filterType == 'forwardFilter':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.filteredObs[i]
                 for i in range(start, end)])
        elif filterType == 'backwardSmoother':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.smoothedObs[i]
                 for i in range(start, end)])
        elif filterType == 'predict':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.predictedObs[i]
                 for i in range(start, end)])
        else:
            raise NameError('Incorrect filter type.')

    def getInterval(self, p=0.95, filterType='forwardFilter', name='main'):
        """ get the confidence interval for data or component.

        If the filtered dates are not
        (0, self.n - 1), then a warning will prompt stating the actual
        filtered dates.

        Args:
            p: The confidence level.
            filterType: the type of CI to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get CI. When name = 'main', then it
                  returns the confidence interval for the time series. When
                  name = some component's name, then it returns the confidence
                  interval for that component. Default to 'main'.

        Returns:
            A tuple with the first element being a list of upper bounds
            and the second being a list of the lower bounds.

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # get the mean and the variance for the time series data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                compMean = self._1DmatrixToArray(
                    self.result.filteredObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.filteredObsVar[start:end])
            elif filterType == 'backwardSmoother':
                compMean = self._1DmatrixToArray(
                    self.result.smoothedObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.smoothedObsVar[start:end])
            elif filterType == 'predict':
                compMean = self._1DmatrixToArray(
                    self.result.predictedObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.predictedObsVar[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the mean and variance for the component
        else:
            self._checkComponent(name)
            compMean = self._getComponentMean(name=name,
                                              filterType=filterType,
                                              start=start, end=(end - 1))
            compVar = self._getComponentVar(name=name,
                                            filterType=filterType,
                                            start=start, end=(end - 1))

        # get the upper and lower bound
        upper, lower = getInterval(compMean, compVar, p)
        return (upper, lower)

    def getLatentState(self, filterType='forwardFilter', name='all'):
        """ get the latent states for different components and filters.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of latent states to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get latent state. When name = 'all', then it
                  returns the latent states for the time series. When
                  name = some component's name, then it returns the latent
                  states for that component. Default to 'all'.

        Returns:
            A list of lists, standing for the latent states given
            the different choices.

        """
        # get the working dates
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # to return the full latent states
        if name == 'all':
            if filterType == 'forwardFilter':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.filteredState[start:end]))
            elif filterType == 'backwardSmoother':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.smoothedState[start:end]))
            elif filterType == 'predict':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.smoothedState[start:end]))
            else:
                raise NameError('Incorrect filter type.')

        # to return the latent state for a given component
        self._checkComponent(name)
        return list(map(lambda x: x if x is None else self._1DmatrixToArray(x),
                        self._getLatentState(name=name, filterType=filterType,
                                             start=start, end=(end - 1))))

    def getLatentCov(self, filterType='forwardFilter', name='all'):
        """ get the error covariance for different components and
        filters.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of latent covariance to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get latent cov. When name = 'all', then it
                  returns the latent covariance for the time series. When
                  name = some component's name, then it returns the latent
                  covariance for that component. Default to 'all'.

        Returns:
            A list of numpy matrices, standing for the filtered latent
            covariance.

        """
        # get the working dates
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # to return the full latent covariance
        if name == 'all':
            if filterType == 'forwardFilter':
                return self.result.filteredCov[start:end]
            elif filterType == 'backwardSmoother':
                return self.result.smoothedCov[start:end]
            elif filterType == 'predict':
                return self.result.smoothedCov[start:end]
            else:
                raise NameError('Incorrect filter type.')

        # to return the latent covariance for a given component
        self._checkComponent(name)
        return self._getLatentCov(name=name, filterType=filterType,
                                  start=start, end=(end - 1))

# ======================= data appending, popping and altering ===============

    # Append new data or features to the dlm
    def append(self, data, component='main'):
        """ Append the new data to the main data or the components (new feature data)

        Args:
            data: the new data
            component: the name of which the new data to be added to.\n
                       'main': the main time series data\n
                       other omponent name: add new feature data to other
                       component.

        """
        # initialize the model to ease the modification
        if not self.initialized:
            self._initialize()
            
        # if we are adding new data to the time series
        if component == 'main':
            # add the data to the self.data
            self.data.extend(list(data))

            # update the length
            self.n += len(data)
            self.result._appendResult(len(data))

            # update the automatic components as well
            for component in self.builder.automaticComponents:
                comp = self.builder.automaticComponents[component]
                comp.appendNewData(data)

            # give a warning to remind to append dynamic components
            if len(self.builder.dynamicComponents) > 0:
                print('Remember to append the new features for the' +
                      ' dynamic components as well')

        # if we are adding new data to the features of dynamic components
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
            raise NameError('The date should be between 0 and ' +
                            str(self.n - 1))

        # initialize the model to ease the modification
        if not self.initialized:
            self._initialize()

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
    def alter(self, date, data, component='main'):
        """ To alter the data for a specific date and a specific component.

        Args:
            date: the date of the altering data
            data: the new data. data must be a numeric value for main time
                  series and must be a list of numerical values for dynamic
                  components.
            component: the component for which the new data need to be
                       supplied to.\n
                       'main': the main time series data\n
                       other component name: other component feature data

        """
        if date < 0 or date > self.n - 1:
            raise NameError('The date should be between 0 and ' +
                            str(self.n - 1))

        # initialize the model to ease the modification
        if not self.initialized:
            self._initialize()

        # to alter the data for the observed chain
        if component == 'main':
            self.data[date] = data

            # we also automatically alter all the automatic components
            for component in self.builder.automaticComponents:
                comp = self.builder.automaticComponents[component]
                comp.alter(date, data)

        # to alter the feature of a component
        elif component in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[component]
            comp.alter(date, data)

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
            raise NameError('The date should be between 0 and ' +
                            str(self.n - 1))

        self.alter(date=date, data=None, component='main')

# ============================= plot component =========================

    def turnOn(self, switch):
        """ "turn on" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch on. \n
                    'filtered plot', 'filter' to plot filtered results\n
                    'smoothed plot', 'smooth' to plot smoothed results\n
                    'predict plot', 'predict', to plot one-step ahead results\n
                    'confidence interval', 'confierence', 'CI' to plot CI's\n
                    'data points', 'data', 'data point' to plot original data\n
                    'multiple', 'separate' to plot results in separate
                    figures\n
                    'fitted dots', 'fitted' to plot fitted results with dots
        """
        if switch in set(['filtered plot', 'filter',
                          'filtered results', 'filtering']):
            self.options.plotFilteredData = True
        elif switch in set(['smoothed plot', 'smooth',
                            'smoothed results', 'smoothing']):
            self.options.plotSmoothedData = True
        elif switch in set(['predict plot', 'predict',
                            'predicted results', 'prediction']):
            self.options.plotPredictedData = True
        elif switch in set(['confidence interval', 'confidence',
                            'interval', 'CI', 'ci']):
            self.options.showConfidenceInterval = True
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = True
        elif switch in set(['multiple', 'multiple plots',
                            'separate plots', 'separate']):
            self.options.separatePlot = True
        elif switch in set(['fitted dots', 'fitted results',
                            'fitted data', 'fitted']):
            self.options.showFittedPoint = True
        else:
            raise NameError('no such options')

    def turnOff(self, switch):
        """ "turn off" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch off. \n
                    'filtered plot', 'filter' to not plot filtered results\n
                    'smoothed plot', 'smooth' to not plot smoothed results\n
                    'predict plot', 'predict', to not plot one-step ahead
                    results\n
                    'confidence interval', 'confierence', 'CI'
                    to not plot CI's\n
                    'data points', 'data', 'data point' to not
                    plot original data\n
                    'multiple', 'separate' to not plot results
                    in separate figures\n
                    'fitted dots', 'fitted' to not plot fitted
                    results with dots

        """
        if switch in set(['filtered plot', 'filter', 'filtered results',
                          'filtering']):
            self.options.plotFilteredData = False
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results',
                            'smoothing']):
            self.options.plotSmoothedData = False
        elif switch in set(['predict plot', 'predict', 'predicted results',
                            'prediction']):
            self.options.plotPredictedData = False
        elif switch in set(['confidence interval', 'confidence', 'interval',
                            'CI', 'ci']):
            self.options.showConfidenceInterval = False
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = False
        elif switch in set(['multiple', 'multiple plots', 'separate plots',
                            'separate']):
            self.options.separatePlot = False
        elif switch in set(['fitted dots', 'fitted results', 'fitted data',
                            'fitted']):
            self.options.showFittedPoint = False
        else:
            raise NameError('no such options')

    def setColor(self, switch, color):
        """ "set" Operation for the dlm plotting colors

        Args:
            switch: key word. Controls over
                    filtered/smoothed/predicted results,
            color: the color for the corresponding keyword.
        """
        if switch in set(['filtered plot', 'filter', 'filtered results',
                          'filtering']):
            self.options.filteredColor = color
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results',
                            'smoothing']):
            self.options.smoothedColor = color
        elif switch in set(['predict plot', 'predict', 'predicted results',
                            'prediction']):
            self.options.predictedColor = color
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.dataColor = color
        else:
            raise NameError('no such options')

    def setConfidence(self, p=0.95):
        """ Set the confidence interval for the plot

        """
        assert p >= 0 and p <= 1
        self.options.confidence = p

    def setIntervalType(self, intervalType):
        """ Set the confidence interval type

        """
        if intervalType == 'ribbon' or intervalType == 'line':
            self.options.intervalType = intervalType
        else:
            raise NameError('No such type for confidence interval.')

    def resetPlotOptions(self):
        """ Reset the plotting option for the dlm class

        """
        self.options.plotOriginalData = True
        self.options.plotFilteredData = True
        self.options.plotSmoothedData = True
        self.options.plotPredictedData = True
        self.options.showDataPoint = False
        self.options.showFittedPoint = False
        self.options.showConfidenceInterval = True
        self.options.dataColor = 'black'
        self.options.filteredColor = 'blue'
        self.options.predictedColor = 'green'
        self.options.smoothedColor = 'red'
        self.options.separatePlot = True
        self.options.confidence = 0.95
        self.options.intervalType = 'ribbon'

    # plot the result according to the options
    def plot(self, name='main'):
        """ The main plot function. The dlmPlot and the matplotlib will only be loaded
        when necessary.

        Args:
            name: component to plot. Default to 'main', in which we plot the
                  filtered time series. If a component name is given
                  It plots the mean of the component, i.e., the observed value
                  that attributes to that particular component, which equals to
                  evaluation * latent states for that particular component.

        """

        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # initialize the figure
        dlmPlot.plotInitialize()

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        # plot the main time series after filtering
        if name == 'main':
            # if we just need one plot
            if self.options.separatePlot is not True:
                dlmPlot.plotInOneFigure(time=time,
                                        data=self.data,
                                        result=self.result,
                                        options=self.options)
            # otherwise, we plot in multiple figures
            else:
                dlmPlot.plotInMultipleFigure(time=time,
                                             data=self.data,
                                             result=self.result,
                                             options=self.options)

        # plot the component after filtering
        elif self._checkComponent(name):
            # create the data for ploting
            data = {}
            if self.options.plotFilteredData:
                data['filteredMean'] = self.getMean(
                    filterType='forwardFilter', name=name)
                data['filteredVar'] = self.getVar(
                    filterType='forwardFilter', name=name)

            if self.options.plotSmoothedData:
                data['smoothedMean'] = self.getMean(
                    filterType='backwardSmoother', name=name)
                data['smoothedVar'] = self.getVar(
                    filterType='backwardSmoother', name=name)

            if self.options.plotPredictedData:
                data['predictedMean'] = self.getMean(
                    filterType='predict', name=name)
                data['predictedVar'] = self.getVar(
                    filterType='predict', name=name)

            if len(data) == 0:
                raise NameError('Nothing is going to be drawn, due to ' +
                                'user choices.')
            data['name'] = name
            dlmPlot.plotComponent(time=time,
                                  data=data,
                                  result=self.result,
                                  options=self.options)
        dlmPlot.plotout()

    def plotCoef(self, name, dimensions=None):
        """ Plot function for the latent states (coefficents of dynamic
        component).

        Args:
            name: the name of the component to plot.
                  It plots the latent states for the component. If dimension of
                  the given component is too high, we truncate
                  to the first five. Or the user can supply the ideal
                  dimensions for plot in the dimensions parameter.
            dimensions: dimensions will be used
                        as the indexes to plot within that component latent
                        states.
        """
        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        # provide a fake time for plotting
        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        # plot the latent states for a given component
        if self._checkComponent(name):

            # find its coordinates in the latent state
            indx = self.builder.componentIndex[name]
            # get the real latent states
            coordinates = range(indx[0], (indx[1] + 1))
            # if user supplies the dimensions
            if dimensions is not None:
                coordinates = [coordinates[i] for i in dimensions]

            # otherwise, if there are too many latent states, we
            # truncated to the first five.
            elif len(coordinates) > 5:
                coordinates = coordinates[:5]

            dlmPlot.plotLatentState(time=time,
                                    coordinates=coordinates,
                                    result=self.result,
                                    options=self.options,
                                    name=name)
        else:
            raise NameError('No such component.')

        dlmPlot.plotout()

    def plotPredictN(self, N=1, date=None, featureDict=None):
        """
        Function to plot the N-day prediction results.

        The input is the same as `dlm.predictN`. For details,
        please refer to that function.

        Args:
            N:    The length of days to predict.
            date: The index when the prediction based on. Default to the
                  last day.
            FeatureDict: The feature set for the dynamic Components, in a form
                  of {"component_name": feature}, where the feature must have
                  N elements of feature vectors. If the featureDict is not
                  supplied, then the algo reuses those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.
        """
        if date is None:
            date = self.n - 1

        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        # provide a fake time for plotting
        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        predictedTimeRange = range(date, date + N)
        predictedData, predictedVar = self.predictN(
            N=N, date=date, featureDict=featureDict)
        dlmPlot.plotPrediction(
            time=time, data=self.data,
            predictedTime=predictedTimeRange,
            predictedData=predictedData,
            predictedVar=predictedVar,
            options=self.options)

        dlmPlot.plotout()
                               
# ================================ control options =========================

    def showOptions(self):
        """ Print out all the option values

        """
        allItems = vars(self.options)
        for item in allItems:
            print(item + ': ' + str(allItems[item]))

    def stableMode(self, use=True):
        """ Turn on the stable mode, i.e., using the renewal strategy.

            Indicate whether the renew strategy should be used to add numerical
            stability. When the filter goes over certain steps,
            the information contribution of the previous data has decayed
            to minimum. In the stable mode, We then ignore those days and
            refit the time series starting from current - renewTerm, where
            renewTerm is computed according to the discount. Thus,
            the effective sample size of the dlm is twice
            renewTerm. When discount = 1, there will be no renewTerm,
            since all the information will be passed along.
        """
        # if option changes, reset everything
        if self.options.stable != use:
            self.initialized = False

        if use is True:
            self.options.stable = True
        elif use is False:
            self.options.stable = False
        else:
            raise NameError('Incorrect option input')

    def evolveMode(self, evoType='dependent'):
        """ Control whether different component evolve indpendently. If true,
        then the innovation will only be added on each component but not the
        correlation between the components, so that for component with discount
        equals to 1, the smoothed results will always be constant.

        Args:
            evoType: If set to 'independent', then each component will evolve
                     independently. If set to 'dependent', then the components
                     will proceed jointly. Default to 'independent'. Switch to
                     'dependent' if efficiency is a concern.

        Returns:
            a dlm object (for chaining purpose)
        """
        # if option changes, reset everything
        if (self.options.innovationType == 'whole' and
            evoType == 'independent') or \
           (self.options.innovationType == 'component' and
            evoType == 'dependent'):
            self.initialized = False

        if evoType == 'independent':
            self.options.innovationType = 'component'
        elif evoType == 'dependent':
            self.options.innovationType = 'whole'
        else:
            raise NameError('Incorrect option input')

        # for chaining
        return self

    def noisePrior(self, prior=0):
        """ To set the prior for the observational noise. Calling with empty
        argument will enable the auto noise intializer (currently, the min of 1
        and the variance of time series).

        Args:
            prior: the prior of the observational noise.

        Returns:
            A dlm object (for chaining purpose)
        """
        if prior > 0:
            self.options.noise=prior
            self.initialized = False
        else:
            self.options.useAutoNoise = True
            self.initialized = False            

        # for chaining
        return self

    def loadPlotLibrary(self):
        if not self.plotLibLoaded:
            global dlmPlot
            import pydlm.plot.dlmPlot as dlmPlot
            self.plotLibLoaded = True
            
# ========================= tuning and evaluation =========================
    def getMSE(self):
        """ Get the one-day ahead prediction mean square error. The mse is
        estimated only for days that has been predicted.

        Returns:
            An numerical value
        """

        return self._getMSE()

    def tune(self, maxit=100):
        """ Automatic tuning of the discounting factors. 

        The method will call the model tuner class to use the default parameters
        to tune the discounting factors and change the discount factor permenantly.
        User needs to refit the model after tuning.
        
        If user wants a more refined tuning and not change any property of the
        existing model, they should opt to use the @modelTuner class.
        """
        simpleTuner = modelTuner()

        if self._printInfo:
            self.fitForwardFilter()
            print("The current mse is " + str(self.getMSE()) + '.')
        
        simpleTuner.tune(untunedDLM=self, maxit=maxit)
        self._setDiscounts(simpleTuner.getDiscounts(), change_component=True)
        
        if self._printInfo:
            self.fitForwardFilter()
            print("The new mse is " + str(self.getMSE()) + '.')
