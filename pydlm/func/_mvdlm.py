"""
=====================================================================

This implements a multivariate DLM

=====================================================================

This code implements a multivariate DLM based on the univariate dlm
structure.

"""
from copy import deepcopy
from time import time
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
        self.initialized = False

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

        for name in self.dlms:
            self.dlms[name].__add__(component)

        return self

    def ls(self):
        if self.dlmType != 'homogeneity':
            raise NameError('ls can only be used for homogeneity mvdlm.' +
                            ' For heterogeneity mvdlm, please use ls' +
                            ' of each dlm itself to see the components')
        name = self.dlms.keys()[0]
        self.dlms[name].ls()

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

# ============================ hiden helper functions =========================
    # check if the data is truely multivariate
    def _checkMultivariate(self, data):
        if not all(isinstance(item, list) for item in data):
            raise NameError('This is not multivariate data.' +
                            ' Use the univariate dlm instead.')

    # copy the data or filtered result to a dlm as its feature.
    def _copyToFeatures(self, name, which):
        current = self.dlms[name]

        # if the feature has not been created,
        # we need to first add it as a feature
        if 'mvdlmFeatures' not in current.dynamicComponents:
            # initialize the new feature
            newFeature = [[0] * len(self.order) for i in range(self.n)]
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
                    if which == 'forwardFilter':
                        theFeature[i][count] = self.dlms[otherdlm] \
                                                   .result.filteredObs[i]
                    elif which == 'backwardSmoother':
                        theFeature[i][count] = self.dlms[otherdlm] \
                                                   .result.smoothedObs[i]
                    elif which == 'original':
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
