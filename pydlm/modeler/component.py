"""
=========================================================================

Code for the abstract component

=========================================================================

This piece of code provide the basic building block for the dynamic linear model.
It provide the fundamental struture for all model components. We implement different
components based on this abstract class.

"""
from abc import ABCMeta, abstractmethod

# We define an abstract class which can further be used
# to create different types of model components, inclusing
# trend, seasonality and other structures


class component:
    """ The abstract class provides the basic structure for all model components
    
    Methods:
        createEvaluation: create the initial evaluation matrix
        createTransition: create the initial transition matrix
        createCovPrior: create a simple prior covariance matrix
        createMeanPrior: create a simple prior latent state
        checkDimensions: if user supplies their own covPrior and meanPrior, this can 
                         be used to check if the dimension matches
    
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """All members that need to be initialized."""
        self.d = None
        self.name = None
        self.componentType = None
        self.discount = None
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

    def __eq__(self, other):
        """Define equal method used for == comparison"""
        if not isinstance(other, component):
            return NotImplemented
        else:
            return (self.equalOrNone(self.d, other.d) and
                    self.equalOrNone(self.name, other.name) and
                    self.equalOrNone(self.componentType, other.componentType) and
                    self.npEqualOrNone(self.discount, other.discount) and
                    self.npEqualOrNone(self.evaluation, other.evaluation) and
                    self.npEqualOrNone(self.transition, other.transition) and
                    self.npEqualOrNone(self.covPrior, other.covPrior) and
                    self.npEqualOrNone(self.meanPrior, other.meanPrior))


    # define the evaluation matrix for the component
    @abstractmethod
    def createEvaluation(self): pass
    """ Create the evaluation matrix

    """


    # define the transition matrix for the component
    @abstractmethod
    def createTransition(self): pass
    """ Create the transition matrix

    """


    # define the prior distribution for the covariance for the component
    @abstractmethod
    def createCovPrior(self): pass
    """ Create the prior covariance matrix for the latent states

    """


    # define the prior distribution for the mean vector for the component
    @abstractmethod
    def createMeanPrior(self): pass
    """ Create the prior latent state
    
    """


    # check the matrix dimensions in case user supplied matrices are wrong
    @abstractmethod
    def checkDimensions(self): pass
    """ Check the dimensionality of the state and covariance

    """
    
    def equalOrNone(self, a, b):
        """Check if a and b are equal or both are None"""
        return (a is None and b is None) or a == b

    def npEqualOrNone(self, a, b):
        """Check if a and b are equal or both are None for NP arrays"""
        return (a is None and b is None) or (a == b).all()
