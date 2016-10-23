"""
=====================================================================

This implements a multivariate DLM

=====================================================================

This code implements a multivariate DLM based on the univariate dlm
structure.

"""
from copy import deepcopy
from time import time
from numpy import matrix
from numpy import transpose
from numpy.linalg import inv
from pydlm import dlm
from pydlm.modeler.dynamic import dynamic


class _mvdlm:

    # we have two ways to initialize a multivariate dlm
    def __init__(self, data=None, dlms=None):

        # if data is provided then this is going to be a
        # homogeneous multivariate dlm, i.e., all the underlying dlm have
        # the same structure.
        if data is None:
            self.dlmType = 'heterogeneity'
        elif len(data) == 0:
            raise NameError('data can not be empty. It can be None type ' +
                            '(indicating this is a heterogeneous mvdlm) ' +
                            'or it must be a list of list containing ' +
                            'multivariate data')
        else:
            self.dlmType = 'homogeneity'
            self._checkMultivariate(data)

        # create the dictionary to store all dlms
        self.dlms = {}

        # store the data length
        self.n = None

        # store the dimension
        self.d = None

        # store the order of the dlms
        self.order = []

        # iteration number
        self.iteration = 5

        # initialization status
        self._initialized = False

        if self.dlmType == 'homogeneity':
            self.n = len(data)
            self.d = len(data[0])
            for i in range(len(data[0])):
                self.dlms[i] = dlm([num[i] for num in data])
                self.dlms[i].printSystemInfo(False)
                self.order.append(i)

        elif self.dlmType == 'heterogeneity' and dlms is not None:
            for name in dlms:
                if self.n is None:
                    self.n = dlms[name].n
                elif self.n != dlms[name].n:
                    raise NameError('The data length for some dlms' +
                                    ' are different.')
                self.dlms[name] = deepcopy(dlms[name])
                self.dlms[name].printSystemInfo(False)
                self.order.append(name)

            self.d = len(self.dlms)

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
    # forward filter for multivariate dlm
    def fitForwardFilter(self,
                         usingRollingWindow=False,
                         windowLength=3,
                         iteration=None):

        # check whether the model has been initialized
        if not self._initialized:
            print ('The multivariate dlm needs to be initialized.')
            self.initialization()

        if iteration is None:
            iteration = self.iteration

        print('Start forward filtering...')
        startTime = time()
        for i in range(iteration):
            print('Total iteration: ' + str(iteration) +
                  '. Current iteration: ' + str(i + 1) + '...')
            for name in self.order:
                self.dlms[name].fitForwardFilter(
                    usingRollingWindow=usingRollingWindow,
                    windowLength=windowLength)

            for name in self.order:
                self._copyToFeatures(name, 'forwardFilter')
            elapsed = time() - startTime
            print('Iteration ' + str(i + 1) + ' completed. Time elapsed: ' +
                  str(elapsed) + '.' + 'Remaining: ' +
                  str(elapsed / (i + 1) * (iteration - i - 1)))

        print ('Forward filtering completed.')

    # backward smoother for multivariate dlm
    def fitBackwardSmoother(self, backLength=None, iteration=None):

        # check whether the model has been initialized
        if not self._initialized:
            raise NameError('You need to run forward filter before ' +
                            'running backward smoother')

        if iteration is None:
            iteration = self.iteration

        print('Start backward smoothing...')
        startTime = time()
        for i in range(iteration):
            print('Total iteration: ' + str(iteration) +
                  '. Current iteration: ' + str(i + 1) + '...')
            for name in self.order:
                self.dlms[name].fitBackwardSmoother(backLength=backLength)

            for name in self.order:
                self._copyToFeatures(name, 'forwardFilter')
            elapsed = time() - startTime
            print('Iteration ' + str(i + 1) + ' completed. Time elapsed: ' +
                  str(elapsed) + '.' + 'Remaining: ' +
                  str(elapsed / (i + 1) * (iteration - i - 1)))

        print ('Backward smoothing completed.')

    # a convenient function using all default settings
    def fit(self):
        self.fitForwardFilter()
        self.fitBackwardSmoother()

# =============================== model prediction ============================
    # the prediction function for forcasting the future
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
    
# ============================ hidden helper functions ========================
    # check if the data is truely multivariate
    def _checkMultivariate(self, data):
        if not all(isinstance(item, list) for item in data):
            raise NameError('This is not multivariate data.' +
                            ' Use the univariate dlm instead.')

    # copy the data or filtered result to a dlm as its feature.
    def _copyToFeatures(self, name, filterType):
        current = self.dlms[name]

        # if the feature has not been created,
        # we need to first add it as a feature
        if 'mvdlmFeatures' not in current.dynamicComponents:
            # initialize the new feature
            newFeature = [[0] * (len(self.order) - 1) for i in range(self.n)]
            current.add(dynamic(features=newFeature,
                                name='mvdlmFeatures',
                                discount=1.0))

        # fetch the features
        theFeature = current.dynamicComponents['mvdlmFeatures'].features

        # update the features
        for i in range(self.n):
            count = 0
            for otherdlm in self.order:
                if otherdlm != name:
                    if filterType == 'forwardFilter':
                        theFeature[i][count] = self.dlms[otherdlm] \
                                                   .result.filteredObs[i]
                    elif filterType == 'backwardSmoother':
                        theFeature[i][count] = self.dlms[otherdlm] \
                                                   .result.smoothedObs[i]
                    elif filterType == 'original':
                        if self.dlms[otherdlm].data[i] is not None:
                            theFeature[i][count] = self.dlms[otherdlm].data[i]
                        else:
                            theFeature[i][count] = 0.0
                    count += 1

    # check whether the dlms contain the same data length
    def _checkDLMLengthAndUpdate(self):

        n = None
        for name in self.dlms:
            if n is None:
                n = self.dlms[name].n
            elif n != self.dlms[name].n:
                return False
        self.n = n
        return True

    # transpose a 2d array
    def _transpose2dArray(self, array2d):
        if array2d is None:
            raise NameError('The 2d array cannot be none.')
        else:
            n = len(array2d)
            if n == 0:
                return []
            m = len(array2d[0])
            if m == 1:
                return array2d
            newArray = [[0] * n for i in range(m)]
            for i in range(n):
                for j in range(m):
                    newArray[j][i] = array2d[i][j]
            return newArray

    # extract the precision and the covariance matrix from mvdlm
    # details are provided in the doc
    def _reconstructPrecision(self, date, filterType):
        precision = matrix([[0] * len(self.order) for i in self.order])
        for i, name in enumerate(self.order):
            unidlm = self.dlms[name]
            precision[i][i] = 1 / unidlm.result.noiseVar[date]
            indx = unidlm.builder.componentIndex['mvdlmFeatures']

            if filterType == 'forwardFilter':                
                coefficient = unidlm.result.filteredState[date][
                    indx[0]:(indx[1] + 1), 0]
            elif filterType == 'backwardSmoother':
                coefficient = unidlm.result.smoothedState[date][
                    indx[0]:(indx[1] + 1), 0]
            elif filterType == 'predicted':
                coefficient = unidlm.result.predictedState[date][
                    indx[0]:(indx[1] + 1), 0]

            for j in range(len(self.order)):
                if j < i:
                    precision[j][i] = - precision[i][i] * \
                                      coefficient[j, 0]
                elif j > i:
                    precision[j][i] = - precision[i][i] * \
                                      coefficient[j - 1, 0]

        return (precision + transpose(precision)) / 2

    # check the type for mvdlm, and raise error for heterogeneity
    # mvdlms. Used for data alternation
    def _checkHeterogeneity(self):
        if self.dlmType == 'heterogeneity':
            raise NameError('For heterogeneous mvdlms, you have to ' +
                            'alter all univariate dlms in their ' +
                            'own level. Any data change in the mvdlm ' +
                            'level only applies to the homogeneity ' +
                            'mvdlms')
