"""
=====================================================================

This implements a multivariate DLM

=====================================================================

This code implements a multivariate DLM based on the univariate dlm
structure.

"""
from copy import deepcopy
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

        if self.dlmType == 'homogeneity':
            self.n = len(data)
            self.d = len(data[0])
            for i in range(len(data[0])):
                self.dlms[i] = dlm([num[i] for num in data])
                self.order.append(i)

        elif self.dlmType == 'heterogeneity' and dlms is not None:
            for name in dlms:
                if self.n is None:
                    self.n = dlms[name].n
                elif self.n != dlms[name].n:
                    raise NameError('The data length for some dlms' +
                                    ' are different.')
                self.dlms[name] = deepcopy(dlms[name])
                self.order.append(name)

            self.d = len(self.dlms)

# =================== modeling function for homogeneity mvdlms ==============

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
        self.order.append(name)
        self.d += 1

    # The initialization function will add to each dlm the other dlms as
    # dynamic features
    def initialize(self):
        for name in self.dlms:
            self._copyToFeatures(name, 'original')

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
                        theFeature[i][count] = self.dlms[otherdlm].data[i]
                count += 1
