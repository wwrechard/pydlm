import numpy as np

# create trend component

class trend:
    def __init__(self, degree = 1):
        if degree <= 0:
            raise NameError('degree has to be positive')
        self.d = degree
        
        # Initialize
        self.evaluation = []
        self.transition = []
        self.covPrior = []
        self.meanPrior = []
        
        # define the evaluation matrix for trend
        self.createEvaluation()

        # define the transition matrix for trend
        self.createTransition()

        # define the prior distribution for the covariance
        self.createCovPrior()

        # define the prior distribution for the mean vector
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
