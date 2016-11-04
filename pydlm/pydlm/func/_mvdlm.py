"""
=====================================================================

This code contains all hidden functions for mvdlm (under development)

=====================================================================

"""
from copy import deepcopy
from time import time
from numpy import matrix
from numpy import transpose
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

        # store the order of the dlms by their names
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

# ============================ hidden helper functions ========================
    # forward filter for multivariate dlm
    def _forwardFilter(self,
                       usingRollingWindow=False,
                       windowLength=3,
                       iteration=None,
                       filterType='forwardFilter'):

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
                self._copyToFeatures(name, filterType)
            elapsed = time() - startTime
            print('Iteration ' + str(i + 1) + ' completed. Time elapsed: ' +
                  str(elapsed) + '.' + 'Remaining: ' +
                  str(elapsed / (i + 1) * (iteration - i - 1)))

        print ('Forward filtering completed.')

    # backward smoother for multivariate dlm
    def _backwardSmoother(self, backLength=None, iteration=None):

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
                self._copyToFeatures(name, 'backwordSmoother')
            elapsed = time() - startTime
            print('Iteration ' + str(i + 1) + ' completed. Time elapsed: ' +
                  str(elapsed) + '.' + 'Remaining: ' +
                  str(elapsed / (i + 1) * (iteration - i - 1)))

        print ('Backward smoothing completed.')

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
