import numpy as np
from component import component
import pydlm.base.tools as tl

# create seasonality component
# We create the seasonality using the component class

class seasonality(component):
    def __init__(self, period = 7):
        if period <= 2:
            raise NameError('Period has to be greater than 1.')
        self.d = period
        
        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createEvaluation()
        self.createTransition()
        self.createCovPrior()
        self.createMeanPrior()

    def createEvaluation(self):
        self.evaluation = np.matrix(np.zeros((1, self.d)))
        self.evaluation[0, 0] = 1

    def createTransition(self):
        self.transition = np.matrix(np.diag(np.ones(self.d - 1), 1))
        self.transition[self.d - 1, 0] = 1
        
    def createCovPrior(self, cov = 1):
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    def freeForm(self):
        if self.covPrior is None or self.meanPrior is None:
            raise NameError('freeForm can only be called after prior created.')
        
    def checkDimensions(self):
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
