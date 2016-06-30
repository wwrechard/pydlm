# This code take care of the Kalman filter
import numpy as np
import tools as tl
import baseModel as bm

# Define the class of Kalman filter which offers a forward filter
# backward smoother and backward sampler for one-step move

class kalmanFilter:

    # One parameter for Kalman filter
    def __init__(self, discount):
        self.discount = discount
        if discount < 0 or discount > 1:
            raise tl.matrixErrors('discount factor must be between 0 and 1')

    # The forward filter of one-step update given a new observation
    def forwardFilter(self, model, y):

        # when y is not a missing data
        if y != 'na':

            # predicted states and obs and their variances
            predState = np.dot(model.transition, model.state)
            predObs = np.dot(model.evaluation, predState)
            predSysVar = np.dot(np.dot(model.transition, model.sysVar), \
                                model.transition.T) + model.innovation
            predObsInvVar = np.dot(np.dot(model.evalution, predSysVar), \
                                model.evalution.T) + model.obsVar
            
            # the prediction error and the correction matrix
            err = y - predObs
            correction = np.dot(predSysVar, model.evaluation.T) / predObsInvVar

            # update new staets
            model.df += 1
            model.obsVar = model.obsVar + 
