import numpy as np
from component import component
import pydlm.base.tools as tl

# create trend component
# We create the trend using the component class

class trend(component):
    def __init__(self, degree = 1, name = 'trend', discount = 0.99):
        if degree <= 0:
            raise NameError('degree has to be positive')
        self.d = degree
        self.type = 'trend'
        self.name = name
        self.discount = np.ones(self.d) * discount
        
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
        self.transition = np.matrix(np.zeros((self.d, self.d)))
        self.transition[np.triu_indices(self.d)] = 1
        
    def createCovPrior(self, cov = 1):
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    def checkDimensions(self):
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print 'The dimesnion looks good!'
