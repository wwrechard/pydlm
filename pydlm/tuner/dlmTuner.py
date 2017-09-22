"""
===============================================================

The discount factor tuner for dlm

===============================================================

The modelTuner class provides the tuning functionality for the dlm class.
It makes use of the gradient descent to optimize the discount factor for
each component (jointly) based on the one-day ahead prediction error.

>>> import modelTuner
>>> myTuner = modelTuner()
>>> tunedDLM = myTuner(untunedDLM, maxit=100)

The tunedDLM will be saved in tunedDLM while the untunedDLM remains unchangd.
An alternative way to call this class is via the tuner method within dlm class.

>>> mydlm.tune(maxit=100)

This will permenantly change the discouting factor in mydlm. So if the user
prefer to build a new dlm with the new discount factor without changing the 
original one, one should opt to use the modelTuner class.

"""
from copy import deepcopy
from numpy import array

class modelTuner:
    """ The main class for modelTuner

    Attributes:
        method: the optimization method. Currently only 'gradient_descent'
                is supported.
        loss:   the optimization loss function. Currently only 'mse' (one-day
                ahead prediction) is supported.
    
    """

    def __init__(self, method='gradient_descent', loss='mse'):

        self.method = method
        self.loss = loss
        self.current_mse = None
        self.err = 1e-4
        self.discounts = None

    def tune(self, untunedDLM, maxit=100, step = 1.0):
        """ Main function for tuning the DLM model.

        Args:
            untunedDLM: The DLM object that needs tuning
            maxit: The maximum number of iteractions for gradient descent.
            step: the moving length at each iteraction.

        Returns:
            A tuned DLM object in unintialized status.
        """
        # make a deep copy of the original dlm
        tunedDLM = deepcopy(untunedDLM)
        tunedDLM.showInternalMessage(False)

        if not tunedDLM.initialized:
            tunedDLM.fitForwardFilter()
        discounts = array(tunedDLM._getDiscounts())
        self.current_mse = tunedDLM._getMSE()
        
        # using gradient descent
        if self.method == 'gradient_descent':
            for i in range(maxit):
                gradient = self.find_gradient(discounts, tunedDLM)
                discounts -= gradient * step
                discounts = list(map(lambda x: self.cutoff(x), discounts))
                tunedDLM._setDiscounts(discounts)
                tunedDLM.fitForwardFilter()
                self.current_mse = tunedDLM._getMSE()

            if i < maxit - 1:
                print('Converge successfully!')
            else:
                print('The algorithm stops without converging.')
                if min(discounts) <= 0.7 + self.err or max(discounts) >= 1 - 2 * self.err:
                    print('Possible reason: some discount is too close to 1 or 0.7' +
                          ' (0.7 is smallest discount that is permissible.')
                else:
                    print('It might require more step to converge.' +
                          ' Use tune(..., maixt = <a larger number>) instead.')

        self.discounts = discounts
        tunedDLM._setDiscounts(discounts, change_component=True)
        return tunedDLM

    def getDiscounts(self):
        """ Get the tuned discounting factors. One for each component (even the
            component being multi-dimensional, only one discounting factor will
            be assigned to one component). Initialized to None.

        """
        return self.discounts

    def find_gradient(self, discounts, DLM):
        if self.current_mse is None:
            self.current_mse = DLM._getMSE()

        gradient = array([0.0] * len(discounts))

        for i in range(len(discounts)):
            discounts_err = discounts
            discounts_err[i] = self.cutoff(discounts_err[i] + self.err)

            DLM._setDiscounts(discounts_err)
            DLM.fitForwardFilter()
            gradient[i] = (DLM._getMSE() - self.current_mse) / self.err

        return gradient

    def cutoff(self, a):
        if a < 0.7:
            return 0.7

        if a >= 1:
            return 0.99999

        return a
