"""
=========================================================================

Code for the dynamic component

=========================================================================

This piece of code provide one building block for the dynamic linear model.
It decribes a dynamic component in the time series data. It basically allows
user to supply covariate or controlled variable into the dlm,
and the coefficients of the features will be trained as the latent states.
Examples are holiday indicators, other observed variables and so on.

The name dynamic means that the features are changing over time.

"""
import numpy as np
from collections import MutableSequence
from copy import deepcopy

import pydlm.base.tools as tl
from .component import component
# create trend component
# We create the trend using the component class

class dynamic(component):
    """ The dynamic component that allows user to add controlled variables,
    providing one building block for the dynamic linear model.
    It decribes a dynamic component in the time series data. It basically allows
    user to supply covariate or controlled variable into the dlm,
    and the coefficients of the features will be trained as the latent states.
    Examples are holiday indicators, other observed variables and so on.

    Args:
        features: the feature matrix of the dynamic component
        discount: the discount factor
        name: the name of the dynamic component
        w: the value to set the prior covariance. Default to a diagonal
           matrix with 1e7 on the diagonal.

    Examples:
        >>>  # create a dynamic component:
        >>> features = [[1.0, 2.0] for i in range(10)]
        >>> ctrend = dynamic(features = features, name = 'random', discount = 0.99)
        >>>  # change the ctrend to have covariance with diagonals are 2 and state 1
        >>> ctrend.createCovPrior(cov = 2)
        >>> ctrend.createMeanPrior(mean = 1)

    Attributes:
        d: the dimension of the features (number of latent states)
        n: the number of observation
        componentType: the type of the component, in this case, 'dynamic'
        name: the name of the trend component, to be supplied by user
              used in modeling and result extraction
        discount: the discount factor for this component. Details please refer
                  to the @kalmanFilter
        evaluation: the evaluation matrix for this component
        transition: the transition matrix for this component
        covPrior: the prior guess of the covariance matrix of the latent states
        meanPrior: the prior guess of the latent states

    """
    def __init__(self,
                 features = None,
                 discount = 0.99,
                 name = 'dynamic',
                 w=100):

        self.n = len(features)
        self.d = len(features[0])

        if self.hasMissingData(features):
            raise NameError("The current version does not support missing data" +
                            "in the features.")

        self.features = deepcopy(features)
        if isinstance(features, np.matrix):
            self.features = self.features.tolist()
        self.componentType = 'dynamic'
        self.name = name
        self.discount = np.ones(self.d) * discount

        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createEvaluation(0)
        self.createTransition()
        self.createCovPrior(scale=w)
        self.createMeanPrior()

        # record current step in case of lost
        self.step = 0

    def createEvaluation(self, step):
        """ The evaluation matrix for the dynamic component change over time.
        It equals to the value of the features or the controlled variables at a
        given date

        """
        self.evaluation = np.matrix([self.features[step]])

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

    # Recursively heck if there is any none data. We currently don't support
    # missing data for features.
    def hasMissingData(self, features):
        """ Check whether the list contains None

        """
        for item in features:
            if isinstance(item, MutableSequence):
                if self.hasMissingData(item):
                    return True
            else:
                if item is None:
                    return True
        return False

    def updateEvaluation(self, step):
        """ update the evaluation matrix to a specific date
        This function is used when fitting the forward filter and backward smoother
        in need of updating the correct evaluation matrix

        """
        if step < self.n:
            self.evaluation = np.matrix([self.features[step]])
            self.step = step
        else:
            raise NameError('The step is out of range')

    def appendNewData(self, newData):
        """ For updating feature matrix when new data is added.

        Args:
            newData: is a list of list. The inner list is the feature vector. The outer
                     list may contain multiple feature vectors.

        """
        if self.hasMissingData(newData):
            raise NameError("The current version does not support missing data" +
                            "in the features.")
        
        self.features.extend(tl.duplicateList(newData))
        self.n = len(self.features)

    def popout(self, date):
        """ For deleting the feature data of a specific date.

        Args:
            date: the index of which to be deleted.

        """
        self.features.pop(date)
        self.n -= 1

    def alter(self, date, feature):
        """ Change the corresponding
            feature matrix.

        Args:
           date: The date to be modified.
           dataPoint: The new feature to be filled in.

        """
        if self.hasMissingData(feature):
            raise NameError("The current version does not support missing data" +
                            "in the features.")
        else:
            self.features[date] = feature

