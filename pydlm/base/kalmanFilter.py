"""
===============================================

Base code for running Kalman filter

===============================================

This module implements a Kalman filter that is slightly different fromt the 
standard one, following West and Harrison (1999). This Kalman filter accepts
one-dimension discounting factor to adaptively learn the innovation matrix
itself, instead of accepting it from the user. (Although such option is still
provided)

"""
# This code take care of the Kalman filter
import numpy as np
import tools as tl

# Define the class of Kalman filter which offers a forward filter
# backward smoother and backward sampler for one-step move

class kalmanFilter:
    """ 
    The kalmanFilter class the provide the basic functionalities

    Members:
        discount: the discounting factor determining how much information to carry on
        updateInnovation: indicate whether the innovation matrix should be updated.
                          default to True.

    Methods:
        predict: predict one step ahead of the current state
        forwardFilter: one step filter on the model given a new observation
        backwardSmoother: one step backward smooth given the future model and the
                          filtered state and systematic covariance
        backwardSampler: similar to backwardSmoother, using sampling instead of 
                         deterministic equations.
        updateDiscount: for updating the discount factors
    """

    def __init__(self, discount = 0.99, updateInnovation = True):
        """ 
        Initializing the kalmanFilter class

        Args:
            discount: the discounting factor, could be a vector
            updateInnovation: the indicator for whether updating innovation matrix

        """
  
        self.__checkDiscount__(discount)
        self.discount = np.matrix(np.diag(1 / np.sqrt(np.array(discount))))
        self.updateInnovation = updateInnovation

 
    def predict(self, model):
        """ 
        Predict the next states of the model by one step

        Args:
            model: the @baseModel class provided all necessary information

        Returns:
            The predicted result is stored in 'model.prediction'

        """
        
        # if the step number == 0, we use result from the model state
        if model.prediction.step == 0:
            model.prediction.state = np.dot(model.transition, model.state)
            model.prediction.obs = np.dot(model.evaluation, model.prediction.state)
            model.prediction.sysVar = np.dot(np.dot(model.transition, model.sysVar), \
                                             model.transition.T)
            # update the innovation
            if self.updateInnovation:
                self.__updateInnovation__(model)

            # add the innovation to the system variance
            model.prediction.sysVar += model.innovation
            model.prediction.obsVar = np.dot(np.dot(model.evaluation, \
                                                    model.prediction.sysVar), \
                                             model.evaluation.T) + model.noiseVar
            model.prediction.step = 1
            
        # otherwise, we use previous result to predict next time stamp 
        else:
            model.prediction.state = np.dot(model.transition, model.prediction.state)
            model.prediction.obs = np.dot(model.evaluation, model.prediction.state)
            model.prediction.sysVar = np.dot(np.dot(model.transition, \
                                                    model.prediction.sysVar),\
                                             model.transition.T)
            model.prediction.obsVar = np.dot(np.dot(model.evaluation, \
                                                    model.prediction.sysVar), \
                                             model.evaluation.T) + model.noiseVar
            model.prediction.step += 1
            

    def forwardFilter(self, model, y):
        """ 
        The forwardFilter used to run one step filtering given new data

        Args:
            model: the @baseModel provided the basic information
            y: the newly observed data

        Returns:
            The filtered result is stored in the 'model' replacing the old states

        """
        
        # when y is not a missing data
        if y is not None:
            
            # first obtain the predicted status
            # we make the prediction step equal to 0 to ensure the prediction
            # is based on the model state and innovation is updated correctlly
            # model.prediction.step = 0
            self.predict(model)
            model.prediction.step = 0
        
            # the prediction error and the correction matrix
            err = y - model.prediction.obs
            correction = np.dot(model.prediction.sysVar, model.evaluation.T) \
                         / model.prediction.obsVar

            # update new staets
            model.df += 1
            lastNoiseVar = model.noiseVar # for updating model.sysVar
            model.noiseVar = model.noiseVar * \
                             (1 - 1 / model.df + \
                              err * err / model.df / model.prediction.obsVar)
            model.state = model.prediction.state + correction * err
            model.sysVar = model.noiseVar[0, 0] / lastNoiseVar[0, 0] * \
                           (model.prediction.sysVar - np.dot(correction, correction.T) * \
                            model.prediction.obsVar[0, 0])
            model.obs = np.dot(model.evaluation, model.state)
            model.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), \
                                  model.evaluation.T) + model.noiseVar
            # update the innovation using discount
            # model.innovation = model.sysVar * (1 / self.discount - 1)

        # when y is missing, then we update the status by the predicted results
        else:
            # we do not update the model.predict.step because we need to take of the case
            # [5, None, None, None]. In such case, we do not add more innovation, because
            # no information is comming in.
            # This is correct because
            # 1. for the first 'None', the step starts from 0 because '5' appears before
            # 2. for the second 'None', the step starts from 1, but the prediction.state
            #    is correct, because now model.state = model.prediciton.state
            # 3. The last 'None' follows the same
            self.predict(model)
            
            model.state = model.prediction.state
            model.sysVar = model.prediction.sysVar
            model.obs = model.prediction.obs
            model.obsVar = model.prediction.obsVar


    # The backward smoother for a given unsmoothed states at time t
    # what model should store:
    #      model.state: the last smoothed states (t + 1)
    #      model.sysVar: the last smoothed system variance (t + 1)
    #      model.transition: the transition at time t + 1
    #      model.evaluation: the evaluation vector at time t
    #      model.prediction.sysVar: the predicted system variance for time t + 1
    #      model.prediction.state: the predicted state for time t + 1
    #      rawState: the unsmoothed state at time t
    #      rawSysVar: the unsmoothed system variance at time t
    def backwardSmoother(self, model, rawState, rawSysVar):
        """ 
        The backwardSmoother for one step backward smoothing

        Args:
            model: the @baseModel used for backward smoothing, the model shall store
                 the following information
                 model.state: the last smoothed states (t + 1)
                 model.sysVar: the last smoothed system variance (t + 1)
                 model.transition: the transition at time t + 1
                 model.evaluation: the evaluation vector at time t
                 model.prediction.sysVar: the predicted system variance for time t + 1
                 model.prediction.state: the predicted state for time t + 1
                 rawState: the unsmoothed state at time t
                 rawSysVar: the unsmoothed system variance at time t
            rawState: the filtered state at the current time stamp
            rawSysVar: the filtered systematic covariance at the current time stamp

        Returns:
            The smoothed results are stored in the 'model' replacing the filtered result.
        """
        
        backward = np.dot(np.dot(rawSysVar, model.transition.T), \
                          np.linalg.inv(model.prediction.sysVar))
        model.state = rawState + np.dot(backward, (model.state - model.prediction.state))
        model.sysVar = rawSysVar + \
                       np.dot(np.dot(backward, \
                                     (model.sysVar - model.prediction.sysVar)), backward.T)
        model.obs = np.dot(model.evaluation, model.state)
        model.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), \
                              model.evaluation.T) + model.noiseVar

    def backwardSampler(self, model, rawState, rawSysVar):
        """ 
        The backwardSampler for one step backward sampling

        Args:
            model: the @baseModel used for backward sampling, the model shall store
                 the following information
                 model.state: the last smoothed states (t + 1)
                 model.sysVar: the last smoothed system variance (t + 1)
                 model.transition: the transition at time t + 1
                 model.evaluation: the evaluation vector at time t
                 model.prediction.sysVar: the predicted system variance for time t + 1
                 model.prediction.state: the predicted state for time t + 1
                 rawState: the unsmoothed state at time t
                 rawSysVar: the unsmoothed system variance at time t
            rawState: the filtered state at the current time stamp
            rawSysVar: the filtered systematic covariance at the current time stamp

        Returns:
            The sampled results are stored in the 'model' replacing the filtered result.
        """
        
        backward = np.dot(np.dot(rawSysVar, model.transition.T), \
                          np.linalg.inv(model.prediction.sysVar))
        model.state = rawState + np.dot(backward, (model.state - model.prediction.state))
        model.sysVar = rawSysVar + \
                       np.dot(np.dot(backward, \
                                     (model.sysVar - model.prediction.sysVar)), backward.T)
        model.state = np.matrix(np.random.multivariate_normal(model.state.A1, \
                                                              model.sysVar)).T
        model.obs = np.dot(model.evaluation, model.state)
        model.obsVar = np.dot(np.dot(model.evaluation, model.sysVar), \
                              model.evaluation.T) + model.noiseVar
        model.obs = np.matrix(np.random.multivariate_normal(model.obs.A1, \
                                                              model.obsVar)).T

    # for updating the discounting factor
    def updateDiscount(self, newDiscount):
        """ 
        For updating the discounting factor

        Args:
            newDiscount: the new discount factor
        """
        
        self.__checkDiscount__(newDiscount)
        self.discount = np.matrix(np.diag(1 / np.sqrt(newDiscount)))
        
    def __checkDiscount__(self, discount):
        """
        Check whether the discount fact is within (0, 1)
        """
        
        for i in range(len(discount)):
            if discount[i] < 0 or discount[i] > 1:
                raise tl.matrixErrors('discount factor must be between 0 and 1')


    # update the innovation
    def __updateInnovation__(self, model):
        """
        update the innovation matrix of the model
        """
        
        model.innovation = np.dot(np.dot(self.discount, model.prediction.sysVar), \
                                      self.discount) - model.prediction.sysVar

