import numpy as np
from component import component
import pydlm.base.tools as tl

# create trend component
# We create the trend using the component class

class dynamic(component):
    def __init__(self, dimension = 1, features = None):
        if dimension <= 0:
            raise NameError('degree has to be positive')
        self.d = dimension
        self.features = features
        self.step = 0
        
        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createEvaluation()
        self.createTransition(self.step)
        self.createCovPrior()
        self.createMeanPrior()

    def createEvaluation(self):
        self.evaluation = np.matrix(np.ones((1, self.d)))

    def createTransition(self, step):
        self.transition = np.matrix(np.diag(self.features[:, step].A1))
        
    def createCovPrior(self, cov = 1):
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    def checkDimensions(self):
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print 'The dimesnion looks good!'
