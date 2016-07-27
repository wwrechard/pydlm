import numpy as np
from component import component
import pydlm.base.tools as tl

# create trend component
# We create the trend using the component class

class dynamic(component):
    def __init__(self, features = None):
        self.d, self.n = features.shape()
        self.features = features.T
        self.step = 0
        self.dynamic = True
        
        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createEvaluation(self.step)
        self.createTransition()
        self.createCovPrior()
        self.createMeanPrior()

    def createEvaluation(self, step):
        self.evaluation = self.features[step, :]

    def createTransition(self):
        self.transition = np.matrix(np.eye(self.d))
        
    def createCovPrior(self, cov = 1):
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    def checkDimensions(self):
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print 'The dimesnion looks good!'

    def updateEvaluation(self):
        self.step += 1
        if self.step < self.n:
            self.evalution = self.features[self.step, :]
            
