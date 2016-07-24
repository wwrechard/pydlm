from abc import ABCMeta, abstractmethod

# We define an abstract class which can further be used
# to create different types of model components, inclusing
# trend, seasonality and other structures

class component:
    __metaclass__ = ABCMeta

    # define the evaluation matrix for the component
    @abstractmethod
    def createEvaluation(self): pass

    # define the transition matrix for the component
    @abstractmethod
    def createTransition(self): pass

    # define the prior distribution for the covariance for the component
    @abstractmethod
    def createCovPrior(self): pass

    # define the prior distribution for the mean vector for the component
    @abstractmethod
    def createMeanPrior(self): pass

    # check the matrix dimensions in case user supplied matrices are wrong
    @abstractmethod
    def checkDimensions(self): pass
