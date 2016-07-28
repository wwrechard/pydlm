import numpy as np
from component import component
import pydlm.base.tools as tl

# create seasonality component
# We create the seasonality using the component class

class seasonality(component):
    def __init__(self, period = 7, discount = 0.99):
        if period <= 2:
            raise NameError('Period has to be greater than 1.')
        self.d = period
        self.dynamic = False
        self.name = 'seasonality'
        self.discount = np.ones(self.d)
        
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

        # create form free seasonality component
        self.freeForm()

    def createEvaluation(self):
        self.evaluation = np.matrix(np.zeros((1, self.d)))
        self.evaluation[0, 0] = 1

    # The transition matrix takes special form as
    # G = [0 1 0]
    #     [0 0 1]
    #     [1 0 0]
    # So everyt time, when G * G, we rotate the vector once, which results
    # in the seasonality performance
    def createTransition(self):
        self.transition = np.matrix(np.diag(np.ones(self.d - 1), 1))
        self.transition[self.d - 1, 0] = 1
        
    def createCovPrior(self, cov = 1):
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    # Form free seasonality component ensures that sum(mean) = 0
    # We use the formular from "Bayesian forecasting and dynamic linear models"
    # Page 242
    def freeForm(self):
        if self.covPrior is None or self.meanPrior is None:
            raise NameError('freeForm can only be called after prior created.')
        else:
            u = np.sum(np.sum(self.covPrior, 0), 1)[0, 0]
            A = np.sum(self.covPrior, 1) / u
            self.meanPrior = self.meanPrior - A * np.sum(self.meanPrior, 0)[0, 0]
            self.covPrior = self.covPrior - np.dot(A, A.T) * u
        
    def checkDimensions(self):
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print 'The dimension looks good!'
