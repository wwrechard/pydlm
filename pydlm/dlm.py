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
from pydlm.predict.dlmPredictMod import dlmPredictModule
from pydlm.access.dlmAccessMod import dlmAccessModule
from pydlm.tuner.dlmTuneMod import dlmTuneModule
from pydlm.plot.dlmPlotMod import dlmPlotModule


class dlm(dlmPlotModule, dlmPredictModule, dlmAccessModule, dlmTuneModule):
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
