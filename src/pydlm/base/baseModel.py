# dependencies
import numpy as np
import tools as tl

# define the basic structure for a dlm model
class baseModel:
            
    # define the components of a baseModel
    def __init__(self, transition = None, evaluation = None, noiseVar = None, \
                 sysVar = None, innovation = None, state = None, df = None):
        self.transition = transition
        self.evaluation = evaluation
        self.noiseVar = noiseVar
        self.sysVar = sysVar
        self.innovation = innovation
        self.state = state
        self.df = df
        self.obs = None
        self.obsVar = None

        # a hidden data field used only for model prediction
        self.prediction = __model__()

    # initialize the observation mean and variance
    def initializeObservation(self):        
        self.validation()
        self.obs = np.dot(self.evaluation, self.state)
        self.obsVar = np.dot(np.dot(self.evaluation, self.sysVar), self.evaluation.T) \
                      + self.noiseVar
        
    # checking if the dimension matches with each other
    def validation(self):

        # check symmetric
        tl.checker.checkSymmetry(self.transition)
        tl.checker.checkSymmetry(self.sysVar)
        if self.innovation is not None:
            tl.checker.checkSymmetry(self.innovation)

        # check wether dimension match
        tl.checker.checkMatrixDimension(self.transition, self.sysVar)
        if self.innovation is not None:
            tl.checker.checkMatrixDimension(self.transition, self.innovation)
        print self.evaluation
        print self.transition
        tl.checker.checkVectorDimension(self.evaluation, self.transition)
        tl.checker.checkVectorDimension(self.state, self.transition)

        
# define an inner class to store intermediate results
class __model__:

    # to store result for prediction
    def __init__(self, step = 0, state = None, obs = None, sysVar = None, obsVar = None):
        self.step = 0
        self.state = state
        self.obs = obs
        self.sysVar = sysVar
        self.obsVar = obsVar
