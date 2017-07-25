"""
=========================================================================

Code for the seasonality component

=========================================================================

This piece of code provide one building block for the dynamic linear model.
It decribes a latent seasonality trending in the time series data. The user
can use this class to construct any periodicy component to the model, for
instance, the hourly, weekly or monthly behavior. Different from the Fourier
series, the seasonality components are nonparametric, i.e., there is no sin
or cos relationship between each state. They can be arbitrarily valued.

"""
import numpy as np
from .component import component
import pydlm.base.tools as tl

# create seasonality component
# We create the seasonality using the component class

class seasonality(component):
    """The seasonality component that features the periodicity behavior,
    providing one building block for the dynamic linear model.
    It decribes a latent seasonality trending in the time series data. The user
    can use this class to construct any periodicy component to the model, for
    instance, the hourly, weekly or monthly behavior. Different from the Fourier
    series, the seasonality components are nonparametric, i.e., there is no sin
    or cos relationship between each state. They can be arbitrarily valued.

    Args:
        period: the period of the
        discount: the discount factor
        name: the name of the trend component
        w: the value to set the prior covariance. Default to a diagonal
           matrix with 100 on the diagonal.

    Examples:
        >>>  # create a 7-day seasonality:
        >>> ctrend = seasonality(period = 7, name = 'weekly', discount = 0.99)
        >>>  # change the ctrend to have covariance with diagonals are 2 and state 1
        >>> ctrend.createCovPrior(cov = 2)
        >>> ctrend.createMeanPrior(mean = 1)
        >>> ctrend.freeForm()

    Attributes:
        d: the period of the seasonality
        componentType: the type of the component, in this case, 'seasonality'
        name: the name of the seasonality component, to be supplied by user
              used in modeling and result extraction
        discount: the discount factor for this component. Details please refer
                  to the @kalmanFilter
        evaluation: the evaluation matrix for this component
        transition: the transition matrix for this component
        covPrior: the prior guess of the covariance matrix of the latent states
        meanPrior: the prior guess of the latent states


    """
    def __init__(self,
                 period = 7,
                 discount = 0.99,
                 name = 'seasonality',
                 w=100):

        if period <= 1:
            raise NameError('Period has to be greater than 1.')
        self.d = period
        self.componentType = 'seasonality'
        self.name = name
        self.discount = np.ones(self.d) * discount

        # Initialize all basic quantities
        self.evaluation = None
        self.transition = None
        self.covPrior = None
        self.meanPrior = None

        # create all basic quantities
        self.createEvaluation()
        self.createTransition()
        self.createCovPrior(cov=w)
        self.createMeanPrior()

        # create form free seasonality component
        self.freeForm()

    def createEvaluation(self):
        """ Create the evaluation matrix

        """
        self.evaluation = np.matrix(np.zeros((1, self.d)))
        self.evaluation[0, 0] = 1

    # The transition matrix takes special form as
    # G = [0 1 0]
    #     [0 0 1]
    #     [1 0 0]
    # So everyt time, when G * G, we rotate the vector once, which results
    # in the seasonality performance
    def createTransition(self):
        """ Create the transition matrix.

        According to Hurrison and West (1999), the transition matrix of seasonality
        takes a form of\n

        [[0 1 0 0],\n
        [0 0 1 0],\n
        [0 0 0 1],\n
        [1 0 0 0]]

        """
        self.transition = np.matrix(np.diag(np.ones(self.d - 1), 1))
        self.transition[self.d - 1, 0] = 1

    def createCovPrior(self, cov = 1e7):
        """Create the prior covariance matrix for the latent states.

        """
        self.covPrior = np.matrix(np.eye(self.d)) * cov

    def createMeanPrior(self, mean = 0):
        """ Create the prior latent state

        """
        self.meanPrior = np.matrix(np.ones((self.d, 1))) * mean

    # Form free seasonality component ensures that sum(mean) = 0
    # We use the formular from "Bayesian forecasting and dynamic linear models"
    # Page 242
    def freeForm(self):
        """ The technique used in Hurrison and West (1999). After calling this method,
        The latent states sum up to 0 and the covariance matrix is degenerate to have
        rank d - 1, so that the sum of the latent states will never change when the
        system evolves

        """
        if self.covPrior is None or self.meanPrior is None:
            raise NameError('freeForm can only be called after prior created.')
        else:
            u = np.sum(np.sum(self.covPrior, 0), 1)[0, 0]
            A = np.sum(self.covPrior, 1) / u
            self.meanPrior = self.meanPrior - A * np.sum(self.meanPrior, 0)[0, 0]
            self.covPrior = self.covPrior - np.dot(A, A.T) * u

    def checkDimensions(self):
        """ if user supplies their own covPrior and meanPrior, this can
        be used to check if the dimension matches

        """
        tl.checker.checkVectorDimension(self.meanPrior, self.covPrior)
        print('The dimension looks good!')
