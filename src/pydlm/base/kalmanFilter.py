# This code take care of the Kalman filter
import numpy as np
import tools as tl

# Define the class of Kalman filter which offers a forward filter
# backward smoother and backward sampler for one-step move

class kalmanFilter:

    # One parameter for Kalman filter
    def __init__(self, discount = np.array([0.99]), updateInnovation = True):
  
        self.__checkDiscount__(discount)
        self.discount = np.matrix(np.diag(1 / np.sqrt(discount)))
        self.updateInnovation = updateInnovation

    # A function used to forecast at n steps
    def predict(self, model, step = 1):

        # clear the previous prediction status
        model.prediction.step = 0

        # start predicting
        for i in range(step):
            self.__predict__(model)
            
    # The forward filter of one-step update given a new observation
    def forwardFilter(self, model, y):

        # first obtain the predicted status
        self.predict(model)
        
        # when y is not a missing data
        if y != 'na':    
            # the prediction error and the correction matrix
            err = y - model.obs
            correction = np.dot(model.prediction.sysVar, model.evaluation.T) \
                         / model.prediciton.obsVar

            # update new staets
            model.df += 1
            lastNoiseVar = model.noiseVar # for updating model.sysVar
            model.noiseVar = model.noiseVar * \
                             (1 - 1 / model.df + err^2 / model.df / model.prediction.obsVar)
            model.state = model.state + correction * err
            model.sysVar = model.noiseVar / lastNoiseVar * \
                           (model.prediction.sysVar - np.dot(correction, correction.T) * \
                            model.prediction.obsVar)
            model.obs = np.dot(model.evaluation, model.state)
            # update the innovation using discount
            # model.innovation = model.sysVar * (1 / self.discount - 1)

        # when y is missing, then we update the status by the predicted results
        else:
            model.state = model.prediction.state
            model.sysVar = model.prediction.sysVar
            model.obs = model.prediction.obs
            model.obsVar = model.prediction.obsVar


    # The backward smoother for a given unsmoothed states at time t
    # what model should store:
    #      model.state: the last smoothed states (t + 1)
    #      model.sysVar: the last smoothed system variance (t + 1)
    #      model.transition: the transition at time t + 1
    #      model.prediction.sysVar: the predicted system variance for time t + 1
    #      model.prediction.state: the predicted state for time t + 1
    #      rawState: the unsmoothed state at time t
    #      rawSysVar: the unsmoothed system variance at time t
    def backwardSmoother(self, model, rawState, rawSysVar):
        backward = np.dot(np.dot(rawSysVar, model.transition.T), \
                          np.linalg.inv(model.prediction.sysVar))
        model.state = rawState + np.dot(backward, (model.state - model.prediction.state))
        model.sysVar = rawSysVar + \
                       np.dot(np.dot(backward, (model.sysVar - model.prediction.sysVar)), \
                              backward.T)
        model.obs = np.dot(model.evaluation, model.state)
        model.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), model.evaluation.T) +\
                       model.noiseVar

    # The backward sampler for a given unsmoothed states at time t
    # what model should store:
    #      model.state: the last smoothed states (t + 1)
    #      model.sysVar: the last smoothed system variance (t + 1)
    #      model.transition: the transition at time t + 1
    #      model.prediction.sysVar: the predicted system variance for time t + 1
    #      model.prediction.state: the predicted state for time t + 1
    #      rawState: the unsmoothed state at time t
    #      rawSysVar: the unsmoothed system variance at time t
    def backwardSampler(self, model, rawState, rawSysVar):
        backward = np.dot(np.dot(rawSysVar, model.transition.T), \
                          np.linalg.inv(model.prediction.sysVar))
        model.state = rawState + np.dot(backward, (model.state - model.prediction.state))
        model.sysVar = rawSysVar + \
                       np.dot(np.dot(backward, (model.sysVar - model.prediction.sysVar)), \
                              backward.T)
        model.state = np.matrix(np.random.multivariate_normal(model.state.A1, \
                                                              model.sysVar)).T
        model.obs = np.dot(model.evaluation, model.state)
        model.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), model.evaluation.T) +\
                       model.noiseVar
        model.obs = np.matrix(np.random.multivariate_normal(model.obs.A1, \
                                                              model.obsVar)).T

    # for updating the discounting factor
    def updateDiscount(self, newDiscount):
        self.__checkDiscount__(newDiscount)
        self.discount = np.matrix(np.diag(1 / np.sqrt(newDiscount)))
        
    def __checkDiscount__(self, discount):
        for i in range(len(discount)):
            if discount[i] < 0 or discount[i] > 1:
                raise tl.matrixErrors('discount factor must be between 0 and 1')
            
    # A hiden method that does one step a head prediction
    def __predict__(self, model):

        # if the step number == 0, we use result from the model state
        if model.prediction.step == 0:
            model.prediction.state = np.dot(model.transition, model.state)
            model.prediciton.obs = np.dot(model.evaluation, model.state)
            model.prediction.sysVar = np.dot(np.dot(model.transition, model.sysVar), \
                                             model.transition.T)
            # update the innovation
            if self.updateInnovation:
                self.__updateInnovation__(model)

            # add the innovation to the system variance
            model.prediction.sysVar += model.innovation
            model.prediction.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), \
                                             model.evaluation.T) + model.noiseVar
            model.prediction.step = 1
            
        # otherwise, we use previous result to predict next time stamp 
        else:
            model.prediction.state = np.dot(model.transition, model.prediction.state)
            model.prediciton.obs = np.dot(model.evaluation, model.prediction.state)
            model.prediction.sysVar = np.dot(np.dot(model.transition, \
                                                    model.prediction.sysVar),\
                                             model.transition.T)
            model.prediction.obsVar = np.dot(np.dot(model.evaluation, \
                                                    model.prediction.sysVar), \
                                             model.evaluation.T) + model.noiseVar
            model.prediction.step += 1

    # update the innovation
    def __updateInnovation__(self, model):
        
        tl.checker.checkMatrixDimension(self.discount, model.transition)
        model.innovation = np.dot(np.dot(self.discount, model.prediction.sysVar), \
                                      self.discount) - model.prediciton.sysVar

