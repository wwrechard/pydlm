"""
=====================================================================

This implements a multivariate DLM (under development)

=====================================================================

This code implements a multivariate DLM based on the univariate dlm
structure.

"""
from copy import deepcopy
from numpy.linalg import inv
from pydlm.func._mvdlm import _mvdlm


class mvdlm(_mvdlm):

    # we have two ways to initialize a multivariate dlm
    def __init__(self, data=None, dlms=None):
        _mvdlm.__init__(self, data, dlms)

# =================== modeling function for homogeneity mvdlm ==============

    # replicate the modeling functionality of dlm for homogeneity mvdlm
    def add(self, component):
        self.__add__(component)

    def __add__(self, component):
        if self.dlmType != 'homogeneity':
            raise NameError('addition can only be used for homogeneity mvdlm' +
                            'For adding dlm to heterogeneity mvdlm, use' +
                            'include')
        if component.name in self.dlms:
            raise NameError('The component must be renamed to be a' +
                            ' different one.')

        for name in self.order:
            self.dlms[name].__add__(component)

        return self

    def ls(self):
        if self.dlmType != 'homogeneity':
            raise NameError('ls can only be used for homogeneity mvdlm.' +
                            ' For heterogeneity mvdlm, please use ls' +
                            ' of each dlm itself to see the components')
        self.dlms[self.order[0]].ls()

    def delete(self, name):
        if self.dlmType != 'homogeneity':
            raise NameError('delete can only be used for homogeneity mvdlm.' +
                            ' For heterogeneity mvdlm, please use delete' +
                            ' of each dlm itself to delete the components')
        for item in self.dlms:
            self.dlms[item].delete(name)

        return self

# ================== modeling function for heterogeneity mvdlm ===============
    def include(self, newDlm, name):
        if self.dlmType != 'heterogeneity':
            raise NameError('include can only be used for heterogeneity' +
                            'mvdlm' +
                            'homogeneity mvdlm has to be construct' +
                            ' simultaneously' +
                            'like a univariate dlm')

        if name in self.dlms:
            raise NameError('The name must be an unused one.')

        if self.n != newDlm.n:
            raise NameError('The data length of the new dlm is inconsistent')

        self.dlms[name] = deepcopy(newDlm)
        self.dlms[name].printSystemInfo(False)
        self.order.append(name)
        self.d += 1
        return self

# =============================== initialization ==============================
    # The initialization function will add to each dlm the other dlms as
    # dynamic features
    def initialize(self):

        print('Begin initialization...')

        for name in self.dlms:
            self._copyToFeatures(name, 'original')
            self.dlms[name]._initialize()

        # check if the length of dlms mathces
        # if so, update self.n to that length
        self._checkDLMLengthAndUpdate()

        print('Initialization complete.')
        self._initialized = True

    # check whether the mvdlm has been initialized
    def isInitialized(self):
        return self._initialized

# =============================== model training ==============================
    # predictive filter for multivariate dlm. This is mainly for predicting
    # purpose. It uses the predicts of each univariate dlm as the feature for
    # the other dlms.
    def fitPredictiveFilter(self,
                            usingRollingWindow=False,
                            windowLength=3,
                            iteration=None):
        self._forwardFilter(usingRollingWindow=usingRollingWindow,
                            windowLength=windowLength,
                            iteration=iteration,
                            filterType='predicted')

    # fit the regular forward filter.
    def fitForwardFilter(self,
                         usingRollingWindow=False,
                         windowLength=3,
                         iteration=None):
        self._forwardFilter(usingRollingWindow=usingRollingWindow,
                            windowLength=windowLength,
                            iteration=iteration,
                            filterType='forwardFilter')

    # backward smoother for multivariate dlm
    def fitBackwardSmoother(self, backLength=None, iteration=None):

        self._backwardSmoother(backLength=backLength, iteration=iteration)

    # a convenient function using all default settings
    def fit(self):
        self.fitForwardFilter()
        self.fitBackwardSmoother()

# =============================== model prediction ============================
    # the prediction function for forcasting the future
    # (TODO: this is wrong, fix it)
    def predict(self, date=None, days=1):
        predictedObs = []
        predictedVar = []
        for name in self.order:
            obs, var = self.dlms[name].predict(date=date, days=days)
            predictedObs.append(obs)
            predictedVar.append(var)
        return (predictedObs, predictedVar)

# =========================== get the result ==================================
    # One can get result directly from each dlms from getDLM
    # Here, we only provide the results that are easily aggregatable.
    # No latent result will be provided on an aggregated level.
    # Get them from each individual dlms

    # list out all univariate dlms
    def getNames(self):
        print('The current model contains following univariate' +
              'dlms, you can get them by their names using getDLMs.')
        print(self.order)
        return self.order

    # get the univariate dlms
    def getDLMs(self, names):
        if not isinstance(names, list):
            if names in self.dlms:
                return deepcopy(self.dlms[names])
            else:
                raise NameError('No such dlm with the name ' + names)
        else:
            dlmlist = []
            for name in names:
                if name in self.dlms:
                    dlmlist.append(deepcopy(self.dlms[name]))
                else:
                    raise NameError('No such dlm with the name ' +
                                    name)
            return dlmlist

    # get the aggregated obs
    def getFilteredObs(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            filteredObs = []
            for dlmName in self.order:
                filteredObs.append(self.dlms[dlmName].getFilteredObs())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(filteredObs)

        elif name in self.dlms:
            return self.dlms[name].getFilteredObs()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated varirance
    def getFilteredVar(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            filteredVar = []
            for dlmName in self.order:
                filteredVar.append(self.dlms[dlmName].getFilteredVar())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(filteredVar)

        elif name in self.dlms:
            return self.dlms[name].getFilteredVar()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated confidence interval
    def getFilteredInterval(self, p, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            filteredUpper = []
            filteredLower = []
            for dlmName in self.order:
                upper, lower = self.dlms[dlmName].getFilteredInterval()
                filteredUpper.append(upper)
                filteredLower.append(lower)
            # we need to tranpose the result to have the correct form
            return (self._transpose2dArray(filteredUpper),
                    self._transpose2dArray(filteredLower))

        elif name in self.dlms:
            return self.dlms[name].getFilteredInterval()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated obs
    def getSmoothedObs(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            smoothedObs = []
            for dlmName in self.order:
                smoothedObs.append(self.dlms[dlmName].getSmoothedObs())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(smoothedObs)

        elif name in self.dlms:
            return self.dlms[name].getSmoothedObs()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated varirance
    def getSmoothedVar(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            smoothedVar = []
            for dlmName in self.order:
                smoothedVar.append(self.dlms[dlmName].getSmoothedVar())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(smoothedVar)

        elif name in self.dlms:
            return self.dlms[name].getSmoothedVar()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated confidence interval
    def getSmoothedInterval(self, p, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            smoothedUpper = []
            smoothedLower = []
            for dlmName in self.order:
                upper, lower = self.dlms[dlmName].getSmoothedInterval()
                smoothedUpper.append(upper)
                smoothedLower.append(lower)
            # we need to tranpose the result to have the correct form
            return (self._transpose2dArray(smoothedUpper),
                    self._transpose2dArray(smoothedLower))

        elif name in self.dlms:
            return self.dlms[name].getSmoothedInterval()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated obs
    def getPredictedObs(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            predictedObs = []
            for dlmName in self.order:
                predictedObs.append(self.dlms[dlmName].getPredictedObs())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(predictedObs)

        elif name in self.dlms:
            return self.dlms[name].getPredictedObs()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated varirance
    def getPredictedVar(self, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            predictedVar = []
            for dlmName in self.order:
                predictedVar.append(self.dlms[dlmName].getPredictedVar())
            # we need to tranpose the result to have the correct form
            return self._transpose2dArray(predictedVar)

        elif name in self.dlms:
            return self.dlms[name].getPredictedVar()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get the aggregated confidence interval
    def getPredictedInterval(self, p, name='all'):
        # if name = all, we return aggregated vector result
        if name == 'all':
            predictedUpper = []
            predictedLower = []
            for dlmName in self.order:
                upper, lower = self.dlms[dlmName].getPredictedInterval()
                predictedUpper.append(upper)
                predictedLower.append(lower)
            # we need to tranpose the result to have the correct form
            return (self._transpose2dArray(predictedUpper),
                    self._transpose2dArray(predictedLower))

        elif name in self.dlms:
            return self.dlms[name].getPredictedInterval()

        else:
            raise NameError('No such dlm or wrong parameter value.')

    # get he precision matrix
    def getPrecision(self, date=None, filterType='backwardSmoother'):
        if date is None:
            date = self.n - 1
        if not self._initialized:
            raise NameError('You need to first train the model!')

        # for filtered and predicted covariance
        if filterType == 'forwardFilter' or filterType == 'predicted':
            # check whether the date is within the filtered range
            if date > self.dlms[self.order[0]].result.filteredSteps[1]:
                raise NameError('The date is not within filtered range')

        # for smoothed covariance
        elif filterType == 'backwardSmoother':
            # check if the date is within the smoothed range
            if date > self.dlms[self.order[0]].result.smoothedSteps[1]:
                raise NameError('The date is outside the smoothed range' +
                                'or the backward smoother has yet to be run')

        # for inappropriate input
        else:
            raise NameError('No such filter type')
                
        return self._reconstructPrecision(date, filterType)

    # get the covariance matrix
    def getCovariance(self, date=None, filterType='backwardSmoother'):
        return inv(self.getPrecision(date, filterType))

# =================== data appending, deleting and altering =================
    # all the alter functionality only works for homogenous mvdlm.
    # For heterogeous mvdlm, user has to do that for each dlms
    # append new data to the model
    def append(self, data, component='main'):
        self._checkHeterogeneity()
        # if we are adding new data to the time series
        if component == 'main':
            # add the data to the self.data
            for i, name in enumerate(self.order):
                newData = [data[j][i] for j in range(len(data))]
                # calls the append to update everything
                self.dlms[name].append(newData)

        # if we are adding new data to the features of dynamic components
        else:
            for name in self.order:
                self.dlms[name].append(data, component)

    def popout(self, date):
        for name in self.order:
            self.dlms[name].popout(date)

    def alter(self, date, data, component='main'):
        # only apply to homogeneous mvdlm
        self._checkHeterogeneity()
        if component == 'main':
            for i, name in enumerate(self.order):
                self.dlms[name].alter(date, data[i], 'main')
            else:
                for name in self.order:
                    self.dlms[name].alter(date, data, component)

    def ignore(self, date):
        for name in self.order:
            self.dlms[name].ignore(date)

# no ploting function for multivariate dlm. Using the individual's
# ploting function instead.
