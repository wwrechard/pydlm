"""
===========================================================================

The code for autoregressive components

===========================================================================

This code implements the autoregressive component as a sub-class of dynamic.
Different from the dynamic component, the features in the autoReg is generated
from the data, and updated according to the data. All other features are
similar to @dynamic.

"""
from numpy import matrix
from warnings import warn
from .component import component

import numpy as np
import pydlm.base.tools as tl


class autoReg(component):
    """ The autoReg class allows user to add an autoregressive component to the dlm.
    This code implements the autoregressive component as a child class of
    component. Different from the dynamic component, the features in the
    autoReg is generated from the data, and updated according to the data.

    The latent states of autoReg are aligned in the order of
    [today - degree, today - degree + 1, ..., today - 2, today - 1]. Thus,
    when fetching the latents from autoReg component, use this order to
    correctly align the coefficients.

    (TODO: change the implementation of autoReg, so that the component uses
     filteredObs as the feature instead of the observed data. To do this, we
     might need pass the dlm as a parameter to autoReg, so that autoReg can
     fetch the filtered observation on the fly)

    Args:
        data: the time series data
        degree: the order of the autoregressive component
        discount: the discount factor
        name: the name of the trend component
        w: the value to set the prior covariance. Default to a diagonal
           matrix with 1e7 on the diagonal.

    Attributes:
        degree: the degree of autoregressive, i.e., how many days to look back
        data: the time series data used for constructing the autoregressive
              features
        discount factor: the discounting factor
        name: the name of the component
        padding: either 0 or None. The number to be padded for the first degree
                 days, as no previous data is observed to form the feature
                 matrix

    """

    def __init__(self,
                 data=None,
                 degree=2,
                 discount=0.99,
                 name='ar2',
                 w=100,
                 padding=0):

        if data is not None:
            warn('The data argument in autoReg is deprecated. Please do not use it.')

        self.componentType = 'autoReg'
        self.d = degree
        self.name = name
        self.discount = np.ones(self.d) * discount
        self.padding = padding

        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createTransition()
        self.createCovPrior(scale=w)
        self.createMeanPrior()

        # record current step in case of lost
        self.step = 0

    def createEvaluation(self, step, data):
        """ The evaluation matrix for auto regressor.

        """
        if step > len(data):
            raise NameError("There is no sufficient data for creating autoregressor.")
        # We pad numbers if the step is too early
        self.evaluation = matrix([[self.padding] * (self.d - step) +
                                  list(data[max(0, (step - self.d)) : step])])

    def createTransition(self):
        """ Create the transition matrix.

        For the dynamic component, the transition matrix is just the identity matrix

        """
        self.transition = np.matrix(np.eye(self.d))

    def createCovPrior(self, cov = None, scale = 1e6):
        """ Create the prior covariance matrix for the latent states

        """
        if cov is None:
            self.covPrior = np.matrix(np.eye(self.d)) * scale
        else:
            self.covPrior = cov * scale

    def createMeanPrior(self, mean = None, scale = 1):
        """ Create the prior latent state

        """
        if mean is None:
            self.meanPrior = np.matrix(np.zeros((self.d, 1))) * scale
        else:
            self.meanPrior = mean * scale

    def checkDimensions(self):
        """ if user supplies their own covPrior and meanPrior, this can
        be used to check if the dimension matches

        """
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print('The dimesnion looks good!')

    def updateEvaluation(self, date, data):
        self.createEvaluation(step=date, data=data)
