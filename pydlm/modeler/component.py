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
