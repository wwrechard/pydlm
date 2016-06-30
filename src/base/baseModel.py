# dependencies
import numpy as np
import tools as tl

# define the basic structure for a dlm model
class baseModel:
    
    # define the components of a baseModel
    def __init__(self, transition, evalution, obsVar, sysVar, innovation, state, df):
        baseModel.transition = transition
        baseModel.evalution = evalution
        baseModel.obsVar = obsVar
        baseModel.sysVar = sysVar
        baseModel.innovation = innovation
        baseModel.state = state
        baseModel.df = df

        self.validation(self)
        baseModel.obs = np.dot(evalution, state)

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
