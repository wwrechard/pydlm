# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

#from pydlm.modeler.builder import builder
from pydlm.func.dlm_func import dlm_base


class dlm(dlm_base):

    # define the basic members
    # initialize the result
    def __init__(self, data):
        dlm_base.__init__(data)

#===================== modeling components =====================

    # add component
    def add(self, component):
        self.__add__(component)

    def __add__(self, component):
        self.builder.__add__(component)
        return self

    # list all components
    def ls(self):
        self.builder.ls()

    # delete one component
    def delete(self, index):
        self.builder.delete(index)

#====================== result components ====================

    def getAll(self):
        return self.result

    def getFilteredObs(self):
        return self.result.filteredObs

    def getFilteredObsVar(self):
        return self.result.filteredObsVar

    def getSmoothedObs(self):
        return self.result.smoothedObs

    def getSmoothedObsVar(self):
        return self.result.smoothedObsVar

    def getPredictedObs(self):
        return self.result.predictedObs

    def getPredictedObsVar(self):
        return self.result.predictedObsVar

    def getFilteredState(self, name = 'all'):
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
