"""
===============================================================================

The code for all hidden method for the class dlm

===============================================================================

This piece of code include all the hidden methods and members of the class dlm.
It provides the basic modeling, filtering, forecasting and smoothing of a dlm.

"""
from numpy import dot
from numpy import var
from pydlm.base.kalmanFilter import kalmanFilter
from pydlm.modeler.builder import builder

# this class defines the basic functionalities for dlm, which is not supposed
# to be used by the user. Most functionality in the main dlm will be
# constructed by using the hidden functions in this class


class _dlm(object):
    """ _dlm includes all hidden functions that used by the class dlm. These hidden
    methods provide the basic modeling, filtering, forecasting and smoothing of
    dlm.

    Attributes:
        data: the observed time series data
        n: the length of the time series data
        result: the inner class that records the filtered and smoothed results
        builder: the @builder that is used for providing the modeling
        functionality. For details please refer to @builder
        Filter: the filter used for filtering time series data using the model.
                For details please refer to @kalmanFilter
        initialized: indicates whether the dynamic linear model has been
                     initialized
        options: model options, including initial guess of the observational
                 variance.
                 More is going to be added (plot options and shrinkage options)
        time: the time label, used for plotting


    Methods:
        _initialize: initialize the dlm (builder and kalmanFilter)
        _forwardFilter: run forward filter for a specific start and end date
        _backwardSmoother: run backward smooth for a specific start and end
                           date
        _predictInSample: predict the latent state and observation for a given
                          period of time (deprecated)
        _oneDayAheadPredict: predict one day a head.
        _continuePredict: continue predicting one day after _oneDayAheadPredict
        _resetModelStatus: reset the model status to its prior status
        _setModelStatus: set the model status to a specific date
        _defaultOptions: a class to store and set default options
        _result: a class to store the results
        _copy: copy the result from the model to the _result class
        _reverseCopy: copy the result from the _result class to the model
        _checkFeatureSize: check whether the features's n matches the data's n
        _checkComponent: check whether a component is in dlm
        _getComponent: get the component if it is in dlm
        _getLatentState: get the latent state for a given component
        _getLatentCov: get the latent covariance for a given component
        _getComponentMean: get the mean of a given component
        _getComponentVar: get the variance of a given component
        _checkPlotOptions: set the correct options according to the fit
        _checkAndGetWorkingDates: get the correct filtering dates
    """
    # define the basic members
    # initialize the result
    def __init__(self, data, **options):

        self.data = list(data)
        # padded_data is used by auto regressor. It is the raw data with missing value
        # replaced by forward filter results. (Missing value include the out of scope
        # predictions.
        self.padded_data = self.data
        self.n = len(data)
        self.result = None
        self.builder = builder()
        self.Filter = None
        self.initialized = False
        self.options = self._defaultOptions(**options)
        self.time = None
        self._printInfo = options.get('printInfo', True)
        self.builder._printInfo = self._printInfo

    # an inner class to store all options
    class _defaultOptions(object):
        """ All plotting and fitting options

        """
        def __init__(self, **kwargs):
            self.noise = kwargs.get('noise', 1.0)
            self.stable = kwargs.get('stable', True)
            self.innovationType = kwargs.get('component', 'component')

            self.plotOriginalData = kwargs.get('plotOriginalData', True)
            self.plotFilteredData = kwargs.get('plotFilteredData', True)
            self.plotSmoothedData = kwargs.get('plotSmoothedData', True)
            self.plotPredictedData = kwargs.get('plotPredictedData', True)
            self.showDataPoint = kwargs.get('showDataPoint', False)
            self.showFittedPoint = kwargs.get('showFittedPoint', False)
            self.showConfidenceInterval = kwargs.get('showConfidenceInterval', True)
            self.dataColor = kwargs.get('dataColor', 'black')
            self.filteredColor = kwargs.get('filteredColor', 'blue')
            self.predictedColor = kwargs.get('predictedColor', 'green')
            self.smoothedColor = kwargs.get('smoothedColor', 'red')
            self.separatePlot = kwargs.get('seperatePlot', True)
            self.confidence = kwargs.get('confidence', 0.95)
            self.intervalType = kwargs.get('intervalType', 'ribbon')
            self.useAutoNoise = kwargs.get('useAutoNoise', False)

    # an inner class to store all results
    class _result(object):
        """ Class to store the results

        """
        # class level (static) variables to record all names
        records = ['filteredObs', 'predictedObs', 'smoothedObs',
                   'filteredObsVar',
                   'predictedObsVar', 'smoothedObsVar', 'noiseVar',
                   'df',
                   'filteredState', 'predictedState', 'smoothedState',
                   'filteredCov', 'predictedCov', 'smoothedCov']

        # quantites to record the result
        def __init__(self, n):

            # initialize all records to be [None] * n
            for variable in self.records:
                setattr(self, variable, [None] * n)

            # record the dates that have been filtered
            self.filteredSteps = [0, -1]
            # record the dates that have been smoothed
            self.smoothedSteps = [0, -1]
            # record the last used filterType
            self.filteredType = None
            # record the current prediction status in the form of
            # [start date, current date, [predictedObs1, predictedObs2,...]]
            self.predictStatus = None
            # One day ahead prediction squared error

        # extend the current record by n blocks
        def _appendResult(self, n):
            for variable in self.records:
                getattr(self, variable).extend([None] * n)

        # pop out a specific date
        def _popout(self, date):
            for variable in self.records:
                getattr(self, variable).pop(date)

    # initialize the builder
    def _initialize(self):
        """ Initialize the model: initialize builder and filter.

        """
        self._autoNoise()
        self.builder.initialize(noise=self.options.noise, data=self.padded_data)
        self.Filter = kalmanFilter(discount=self.builder.discount,
                                   updateInnovation=self.options.innovationType,
                                   index=self.builder.componentIndex)
        self.result = self._result(self.n)
        self.initialized = True

    def _initializeFromBuilder(self, exported_builder):
        self.builder.initializeFromBuilder(data=self.data, exported_builder=exported_builder)
        self.Filter = kalmanFilter(discount=self.builder.discount,
                                   updateInnovation=self.options.innovationType,
                                   index=self.builder.componentIndex)
        self.result = self._result(self.n)
        self.initialized = True

    def _autoNoise(self):
        """ Auto initialize the noise parameter if options.useAutoNoise
        is true.

        """
        if self.options.useAutoNoise:
            trimmed_data = [x for x in self.data if x is not None]
            self.options.noise = min(var(trimmed_data), 1)

    # use the forward filter to filter the data
    # start: the place where the filter started
    # end: the place where the filter ended
    # save: the index for dates where the filtered results should be saved,
    #       could be 'all' or 'end'
    # isForget: indicate where the filter should use the previous state as
    #         prior or just use the prior information from builder
    def _forwardFilter(self,
                       start=0,
                       end=None,
                       save='all',
                       ForgetPrevious=False,
                       renew=False):
        """ Running forwardFilter for the data for a given start and end date

        Args:
            start: the start date
            end: the end date (default to the last day of the chain)
            save: indicate the dates of which the result needs to be saved for.
                  'all' stands for (start, end), otherwise an integer between
                  start and end.
            ForgetPrevious: indicate whether the fitering should start from
                            the prior status or the previous date that has
                            been filtered.
                            (used for rolling window filtering, see @dlm)
            renew: if true, filter will refit certain days when the
                   chain gets
                   too long to add numerical stability, the length of the chain
                   is determined by the information carried on. For example,
                   when discount = 0.9, any days that are 65 days ago together
                   only carry information 1%, so we ignore
                   these days and refit the model to aid stability.
        """
        # the default value for end
        if end is None:
            end = self.n - 1

        # to see if the ff need to run or not
        if start > end:
            return None

        # first we need to initialize the model to the correct status
        # if the start point is 0 or we want to forget the previous result
        if start == 0 or ForgetPrevious:
            self._resetModelStatus()

        # otherwise we use the result on the previous day as the prior
        else:
            if start > self.result.filteredSteps[1] + 1:
                raise NameError('The data before start date has' +
                                ' yet to be filtered! Otherwise set' +
                                ' ForgetPrevious to be True. Check the' +
                                ' <filteredSteps> in <result> object.')
            self._setModelStatus(date=start - 1)

        # we run the forward filter sequentially
        lastRenewPoint = start  # record the last renew point
        for step in range(start, end + 1):

            # first check whether we need to update evaluation or not
            if len(self.builder.dynamicComponents) > 0 or \
               len(self.builder.automaticComponents) > 0:
                self.builder.updateEvaluation(step, self.padded_data)

            # check if rewnew is needed
            if renew and step - lastRenewPoint > self.builder.renewTerm \
               and self.builder.renewTerm > 0.0:
                # we renew the state of the day
                self._resetModelStatus()
                for innerStep in range(step - int(self.builder.renewTerm),
                                       step):
                    self.Filter.forwardFilter(self.builder.model,
                                              self.data[innerStep])
                lastRenewPoint = step

            # then we use the updated model to filter the state
            self.Filter.forwardFilter(self.builder.model, self.data[step])

            # extract the result and record
            if save == 'all' or save == step:
                self._copy(model=self.builder.model,
                           result=self.result,
                           step=step,
                           filterType='forwardFilter')

#        self.result.filteredSteps = (0, end)

    # use the backward smooth to smooth the state
    # start: the last date of the backward filtering chain
    # days: number of days to go back from start
    def _backwardSmoother(self, start=None, days=None, ignoreFuture=False):
        """ Backward smooth over filtered results for a specific start
            and number of days

        Args:
            start: the start date
            days: number of days to be smoothed starting from start towards
                  zero
            ignoreFuture: indicate whether the smoothed should start as if the
                          future data was not observed or using the future data
                          as the initial smoothing status.
        """
        # the default start date is the most recent date
        if start is None:
            start = self.n - 1

        # the default backward days number is the total length
        if days is None:
            end = 0
        else:
            end = max(start - days + 1, 0)

        # the forwardFilter has to be run before the smoother
        if self.result.filteredSteps[1] < start:
            raise NameError('The last day has to be filtered before smoothing! \
            check the <filteredSteps> in <result> object.')

        # and we record the most recent day which does not need to be smooth
        if start == self.n - 1 or ignoreFuture is True:
            self.result.smoothedState[start] = self.result.filteredState[start]
            self.result.smoothedObs[start] = self.result.filteredObs[start]
            self.result.smoothedCov[start] = self.result.filteredCov[start]
            self.result.smoothedObsVar[start] \
                = self.result.filteredObsVar[start]
            self.builder.model.noiseVar = self.result.noiseVar[start]
            start -= 1
        else:
            self.builder.model.noiseVar = self.result.noiseVar[self.n - 1]

        # empty smoothing chain, return None
        if start < end:
            return None

        # insert the previous smoothed dates
        self.builder.model.state = self.result.smoothedState[start + 1]
        self.builder.model.sysVar = self.result.smoothedCov[start + 1]

        # we smooth the result sequantially from start - 1 to end
        dates = list(range(end, start + 1))
        dates.reverse()
        for day in dates:
            # we first update the model to be correct status before smooth
            self.builder.model.prediction.state \
                = self.result.predictedState[day + 1]
            self.builder.model.prediction.sysVar \
                = self.result.predictedCov[day + 1]

            if len(self.builder.dynamicComponents) > 0 or \
               len(self.builder.automaticComponents) > 0:
                self.builder.updateEvaluation(day, self.padded_data)

            # then we use the backward filter to filter the result
            self.Filter.backwardSmoother(
                model=self.builder.model,
                rawState=self.result.filteredState[day],
                rawSysVar=self.result.filteredCov[day])

            # extract the result
            self._copy(model=self.builder.model,
                       result=self.result,
                       step=day,
                       filterType='backwardSmoother')

#        self.result.smoothedSteps = (end, start)

    # Forecast the result based on filtered chains
    def _predictInSample(self, date, days=1):
        """ Predict the model's status based on the model of a specific day

        Args:
            date: the date the prediction is based on
            day: number of days forward that need to be predicted.

        Returns:
            A tuple. (Predicted observation, variance of the predicted
            observation)

        """

        if date + days > self.n - 1:
            raise NameError('The range is out of sample.')

        predictedObs = [None] * days
        predictedObsVar = [None] * days
        # reset the date to the date we are interested in
        self._setModelStatus(date=date)
        self.builder.model.prediction.step = 0
        for i in range(1, days):
            # update the evaluation vector
            if len(self.builder.dynamicComponents) > 0 or \
               len(self.builder.automaticComponents) > 0:
                self.builder.updateEvaluation(date + i, self.padded_data)

            self.Filter.predict(self.builder.model)
            predictedObs[i - 1] = self.builder.model.prediction.obs
            predictedObsVar[i - 1] = self.builder.model.prediction.obsVar

        return (predictedObs, predictedObsVar)

    # Note the following functions will modify the status of the model, so they
    # shall not be directly call through the main model if this behavior is not
    # desired.
    
    # featureDict contains all the features for prediction.
    # It is a dictionary with key equals to the name of the component and
    # the value as the new feature (a list). The function
    # will first use the features provided in this feature dict, if not
    # found, it will fetch the default feature from the component. If
    # it could not find feature for some component, it returns an error
    # The intermediate result will be stored in result.predictStatus as
    # (start_date, next_pred_date, [all_predicted_values]), which will be
    # used by _continuePredict.
    def _oneDayAheadPredict(self, date, featureDict=None):
        """ One day ahead prediction based on the date and the featureDict.
        The prediction could be on the last day and into the future or in 
        the middle of the time series and ignore the rest. For predicting into
        the future, the new features must be supplied to featureDict. For 
        prediction in the middle, the user can still supply the features which
        will be used priorily. The old features will be used if featureDict is
        None.

        Args:
            date: the prediction starts (based on the observation before and
                  on this date)
            featureDict: the new feature value for some dynamic components.
                         must be specified in a form of {component_name: value}
                         if the feature for some dynamic component is not
                         supplied. The algorithm will use the features from
                         the old data. (which means if the prediction is out
                         of sample, then all dynamic component must be provided
                         with the new feature value)

        Returns:
            A tuple of (predicted_mean, predicted_variance)
        """
        if date > self.n - 1:
            raise NameError('The date is beyond the data range.')

        # get the correct status of the model
        self._setModelStatus(date=date)
        self._constructEvaluationForPrediction(
            date=date + 1,
            featureDict=featureDict,
            padded_data=self.padded_data[:(date + 1)])
        
        # initialize the prediction status
        self.builder.model.prediction.step = 0

        # start predicting
        self.Filter.predict(self.builder.model)

        predictedObs = self.builder.model.prediction.obs
        predictedObsVar = self.builder.model.prediction.obsVar
        self.result.predictStatus = [
            date,                   # start_date
            date + 1,               # current_date
            [predictedObs[0, 0]]    # all historical predictions
        ]

        return (predictedObs, predictedObsVar)

    def _continuePredict(self, featureDict=None):
        """ Continue predicting one day after _oneDayAheadPredict or
        after _continuePredict. After using
        _oneDayAheadPredict, the user can continue predicting by using
        _continuePredict. The featureDict act the same as in
        _oneDayAheadPredict.

        Args:
            featureDict: the new feature value for some dynamic components.
                         see @_oneDayAheadPredict

        Returns:
            A tuple of (predicted_mean, predicted_variance)
        """
        if self.result.predictStatus is None:
            raise NameError('_continoousPredict can only be used after ' +
                            '_oneDayAheadPredict')
        startDate = self.result.predictStatus[0]
        currentDate = self.result.predictStatus[1]

        self._constructEvaluationForPrediction(
            date=currentDate + 1,
            featureDict=featureDict,
            padded_data=self.padded_data[:(startDate + 1)] +
                        self.result.predictStatus[2])
        self.Filter.predict(self.builder.model)
        predictedObs = self.builder.model.prediction.obs
        predictedObsVar = self.builder.model.prediction.obsVar
        self.result.predictStatus[1] += 1
        self.result.predictStatus[2].append(predictedObs[0, 0])
        return (predictedObs, predictedObsVar)

    # This function will modify the status of the object, use with caution.
    def _constructEvaluationForPrediction(self,
                                          date,
                                          featureDict=None,
                                          padded_data=None):
        """ Construct the evaluation matrix based on date and featureDict.

        Used for prediction. Features provided in the featureDict will be used
        preferrably. If the feature is not found in featureDict, the algorithm
        will seek it based on the old data and the date.

        Args:
            featureDict: a dictionary containing {dynamic_component_name: value}
                         for update the feature for the corresponding component.
            date: if a dynamic component name is not found in featureDict, the
                  algorithm is using its old feature on the given date.
            padded_data: is the mix of the raw data and the predicted data. It is
                  used by auto regressor.

        """
        # New features are provided. Update dynamic componnet.
        # We distribute the featureDict back to dynamicComponents. If the date is
        # out of bound, we append the feature to the feature set. If the date is
        # within range, we replace the old feature with the new feature.
        if featureDict is not None:
            for name in featureDict:
                if name in self.builder.dynamicComponents:
                    comp = self.builder.dynamicComponents[name]                    
                    # the date is within range
                    if date < comp.n:
                        comp.features[date] = featureDict[name]
                        comp.n += 1
                    elif date < comp.n + 1:
                        comp.features.append(featureDict[name])
                        comp.n += 1
                    else:
                        raise NameError("Feature is missing between the last predicted " +
                                        "day and the new day")
                        
        self.builder.updateEvaluation(date, padded_data)

# =========================== model helper function ==========================

    # to set model to a specific date
    def _setModelStatus(self, date=0):
        """ Set the model status to a specific date (the date mush have been filtered)

        """
        if date < self.result.filteredSteps[0] or \
           date > self.result.filteredSteps[1]:
            raise NameError('The date has yet to be filtered yet. ' +
                            'Check the <filteredSteps> in <result> object.')

        self._reverseCopy(model=self.builder.model,
                          result=self.result,
                          step=date)
        if len(self.builder.dynamicComponents) > 0 or \
           len(self.builder.automaticComponents) > 0:
            self.builder.updateEvaluation(date, self.padded_data)

    # reset model to initial status
    def _resetModelStatus(self):
        """ Reset the model to the prior status

        """
        self.builder.model.state = self.builder.statePrior
        self.builder.model.sysVar = self.builder.sysVarPrior
        self.builder.model.noiseVar = self.builder.noiseVar
        self.builder.model.df = 1
        self.builder.model.initializeObservation()

    # a function used to copy result from the model to the result
    def _copy(self, model, result, step, filterType):
        """ Copy result from the model to _result class

        """

        if filterType == 'forwardFilter':
            result.filteredObs[step] = model.obs
            result.predictedObs[step] = model.prediction.obs
            result.filteredObsVar[step] = model.obsVar
            result.predictedObsVar[step] = model.prediction.obsVar
            result.filteredState[step] = model.state
            result.predictedState[step] = model.prediction.state
            result.filteredCov[step] = model.sysVar
            result.predictedCov[step] = model.prediction.sysVar
            result.noiseVar[step] = model.noiseVar
            result.df[step] = model.df
            # pad missing value with filtered result
            if self.data[step] is None:
                self.padded_data[step] = result.filteredObs[step]

        elif filterType == 'backwardSmoother':
            result.smoothedState[step] = model.state
            result.smoothedObs[step] = model.obs
            result.smoothedCov[step] = model.sysVar
            result.smoothedObsVar[step] = model.obsVar

    def _reverseCopy(self, model, result, step):
        """ Copy result from _result class to the model

        """

        model.obs = result.filteredObs[step]
        model.prediction.obs = result.predictedObs[step]
        model.obsVar = result.filteredObsVar[step]
        model.prediction.obsVar = result.predictedObsVar[step]
        model.state = result.filteredState[step]
        model.prediction.state = result.predictedState[step]
        model.sysVar = result.filteredCov[step]
        model.prediction.sysVar = result.predictedCov[step]
        model.noiseVar = result.noiseVar[step]
        model.df = result.df[step]

    # check if the data size matches the dynamic features
    def _checkFeatureSize(self):
        """ Check features's n matches the data's n

        """
        if len(self.builder.dynamicComponents) > 0:
            for name in self.builder.dynamicComponents:
                if self.builder.dynamicComponents[name].n != self.n:
                    raise NameError('The data size of dlm and '
                                    + name + ' does not match')

    def _1DmatrixToArray(self, arrayOf1dMatrix):
        """ Change an array of 1 x 1 matrix to normal array.

        """
        return [item.tolist()[0][0] for item in arrayOf1dMatrix]

    # function to turn off printing system info
    def _printSystemInfo(self, yes):
        """ Whether the systematic infor should be printed.

        """
        if yes:
            self._printInfo = True
            self.builder._printInfo = True
        else:
            self._printInfo = False
            self.builder._printInfo = False

    # function to judge whether a component is in the model
    def _checkComponent(self, name):
        """ Check whether a component is contained by the dlm.

        Args:
            name: the name of the component

        Returns:
            True or error.
        """
        if name in self.builder.staticComponents or \
           name in self.builder.dynamicComponents or \
           name in self.builder.automaticComponents:
            return True
        else:
            raise NameError('No such component.')

    # function to fetch a component
    def _fetchComponent(self, name):
        """ Get the component if the componeng is in the dlm

        Args:
            name: the name of the component

        Returns:
            The component or error.
        """
        if name in self.builder.staticComponents:
            comp = self.builder.staticComponents[name]
        elif name in self.builder.dynamicComponents:
            comp = self.builder.dynamicComponents[name]
        elif name in self.builder.automaticComponents:
            comp = self.builder.automaticComponents[name]
        else:
            raise NameError('No such component.')
        return comp

    # function to get the corresponding latent state
    def _getLatentState(self, name, filterType, start, end):
        """ Get the latent states of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the latent states to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the latent states to be returned.
            end: the end date to be returned.

        Returns:
            A list of latent states.
        """
        end += 1
        indx = self.builder.componentIndex[name]
        patten = lambda x: x if x is None else x[indx[0]:(indx[1] + 1), 0]

        if filterType == 'forwardFilter':
            return list(map(patten, self.result.filteredState[start:end]))
        elif filterType == 'backwardSmoother':
            return list(map(patten, self.result.smoothedState[start:end]))
        elif filterType == 'predict':
            return list(map(patten, self.result.predictedState[start:end]))
        else:
            raise NameError('Incorrect filter type')

    # function to get the corresponding latent covariance
    def _getLatentCov(self, name, filterType, start, end):
        """ Get the latent covariance of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the latent covariance to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the latent covariance to be returned.
            end: the end date to be returned.

        Returns:
            A list of latent covariance.
        """
        end += 1
        indx = self.builder.componentIndex[name]
        patten = lambda x: x if x is None \
                 else x[indx[0]:(indx[1] + 1), indx[0]:(indx[1] + 1)]

        if filterType == 'forwardFilter':
            return list(map(patten, self.result.filteredCov[start:end]))
        elif filterType == 'backwardSmoother':
            return list(map(patten, self.result.smoothedCov[start:end]))
        elif filterType == 'predict':
            return list(map(patten, self.result.predictedCov[start:end]))
        else:
            raise NameError('Incorrect filter type')

    # function to get the component mean
    def _getComponentMean(self, name, filterType, start, end):
        """ Get the mean of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the mean to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the mean to be returned.
            end: the end date to be returned.

        Returns:
            A list of mean.
        """
        end += 1
        comp = self._fetchComponent(name)
        componentState = self._getLatentState(name=name,
                                              filterType=filterType,
                                              start=start, end=end)
        result = []
        for k, i in enumerate(range(start, end)):
            if name in self.builder.dynamicComponents:
                comp.updateEvaluation(i)
            elif name in self.builder.automaticComponents:
                comp.updateEvaluation(i, self.padded_data)
            result.append(dot(comp.evaluation,
                              componentState[k]).tolist()[0][0])
        return result

    # function to get the component variance
    def _getComponentVar(self, name, filterType, start, end):
        """ Get the variance of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the variance to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the variance to be returned.
            end: the end date to be returned.

        Returns:
            A list of variance.
        """
        end += 1
        comp = self._fetchComponent(name)
        componentCov = self._getLatentCov(name=name,
                                          filterType=filterType,
                                          start=start, end=end)
        result = []
        for k, i in enumerate(range(start, end)):
            if name in self.builder.dynamicComponents:
                comp.updateEvaluation(i)
            elif name in self.builder.automaticComponents:
                comp.updateEvaluation(i, self.padded_data)
            result.append(dot(
                dot(comp.evaluation,
                    componentCov[k]), comp.evaluation.T).tolist()[0][0])
        return result

    # check start and end dates that has been filtered on
    def _checkAndGetWorkingDates(self, filterType):
        """ Check the filter status and return the dates that have
        been filtered according to the filterType.

        Args:
            filterType: the type of the filter to check. Could be
                       "forwardFilter", "backwardSmoother" and "predict"

        Returns:
            a tuple of (start_date, end_date)
        """
        if filterType == 'forwardFilter' or filterType == 'predict':
            if self.result.filteredSteps != [0, self.n - 1] \
               and self._printInfo:
                print('The fitlered dates are from ' +
                      str(self.result.filteredSteps[0]) +
                      ' to ' + str(self.result.filteredSteps[1]))
            start = self.result.filteredSteps[0]
            end = self.result.filteredSteps[1]

        elif filterType == 'backwardSmoother':
            if self.result.smoothedSteps != [0, self.n - 1] \
               and self._printInfo:
                print('The smoothed dates are from ' +
                      str(self.result.smoothedSteps[0]) +
                      ' to ' + str(self.result.smoothedSteps[1]))
            start = self.result.smoothedSteps[0]
            end = self.result.smoothedSteps[1]

        else:
            raise NameError('Incorrect filter type.')

        return (start, end)

    # check the filter status and automatic turn off some plots
    def _checkPlotOptions(self):
        """ Check the filter status and determine the plot options.

        """
        # adjust the option according to the filtering status
        if self.result.filteredSteps[1] == -1:
            self.options.plotFilteredData = False
            self.options.plotPredictedData = False

        if self.result.smoothedSteps[1] == -1:
            self.options.plotSmoothedData = False

#====================== function for discount tuning =========================
    # get the mse from the model
    def _getMSE(self):
        
        if not self.initialized:
            raise NameError('need to fit the model first')

        if self.result.filteredSteps[1] == -1:
            raise NameError('need to run forward filter first')

        mse = 0
        for i in range(self.result.filteredSteps[0],
                       self.result.filteredSteps[1] + 1):
            if self.data[i] is not None:
                mse += (self.data[i] - self.result.predictedObs[i]) ** 2

        mse = mse / (self.result.filteredSteps[1] + 1 -
                      self.result.filteredSteps[0])
        return mse[0,0]

    # get the discount from the model
    def _getDiscounts(self):
        
        if not self.initialized:
            raise NameError('need to fit the model before one can' +
                            'fetch the discount factors')

        discounts = []
        for comp in self.builder.componentIndex:
            indx = self.builder.componentIndex[comp]
            discounts.append(self.builder.discount[indx[0]])
        return discounts

    # set the model discount, this should never expose to the user
    # change the discount in the component would change the whole model.
    # change those in filter and builder only change the discounts
    # temporarily and will be corrected if we rebuild the model.
    def _setDiscounts(self, discounts, change_component=False):

        if not self.initialized:
            raise NameError('need to fit the model first')

        for i, comp in enumerate(self.builder.componentIndex):
            indx = self.builder.componentIndex[comp]
            self.builder.discount[indx[0]: (indx[1] + 1)] = discounts[i]
            if change_component:
                component = self._fetchComponent(name=comp)
                component.discount = self.builder.discount[indx[0]: (indx[1] + 1)]

        self.Filter.updateDiscount(self.builder.discount)
        self.result.filteredSteps = [0, -1]

    # whether to show the internal message
    def showInternalMessage(self, show=True):
        self._printInfo = show
