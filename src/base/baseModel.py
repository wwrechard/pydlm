# dependencies
import numpy as np
import tools as tl

# define the basic structure for a dlm model
class baseModel:
            
    # define the components of a baseModel
    def __init__(self, transition, evaluation, noiseVar, sysVar, innovation, state, df):
        self.transition = transition
        self.evaluation = evaluation
        self.noiseVar = noiseVar
        self.sysVar = sysVar
        self.innovation = innovation
        self.state = state
        self.df = df

        self.validation(self)
        self.obs = np.dot(evaluation, state)
        self.obsVar = np.dot(np.dot(self.evaluation, self.sysVar), self.evalution) \
                      + self.noiseVar

        # a hidden data field used only for model prediction
        self.prediction = self.__model__()

    # checking if the dimension matches with each other
    def validation(self):

        # check symmetric
        tl.checker.checkSymmetry(self.transition)
        tl.checker.checkSymmetry(self.obsVar)
        tl.checker.checkSymmetry(self.sysVar)
        tl.checker.checkSymmetry(self.innovation)

        # check wether dimension match
        tl.checker.checkMatrixDimension(self.transition, self.sysVar)
        tl.checker.checkMatrixDimension(self.transition, self.innovation)
        tl.checker.checkVectorDimension(self.evalution, self.transition)
        tl.checker.checkVectorDimension(self.obsVar, self.evalution)
        tl.checker.checkVectorDimension(self.state, self.transition)

        
# define an inner class to store intermediate results
class __model__:

    # to store result for prediction
    def __init__(self, step = 0, state = [], obs = [], sysVar = [], obsVar = []):
        self.step = 0
        self.state = state
        self.obs = obs
        self.sysVar = sysVar
        self.obsVar = obsVar
