"""
===============================================================

The discount factor tuner for dlm

===============================================================

"""
from copy import deepcopy
from numpy import array

class modelTuner:

    def __init__(self, method='gradient_descent', loss='mse'):

        self.method = method
        self.loss = loss
        self.current_mse = None
        self.err = 1e-4
        self.discounts = None

    def tune(self, untunedDLM, maxit=100, step = 1.0):

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
                discounts = map(lambda x: self.cutoff(x), discounts)
                tunedDLM._setDiscounts(discounts)
                tunedDLM.fitForwardFilter()
                self.current_mse = tunedDLM._getMSE()

            if i < maxit - 1:
                print('Converge successfully!')
            else:
                print('The algorithm stops without converging.')
                if min(discounts) <= 0.1 + self.err or max(discounts) >= 1 - 2 * self.err:
                    print('Possible reason: some discount is too close to 1 or 0.1')
                else:
                    print('It might require more step to converge.' +
                          ' Use tune(..., maixt = <a larger number>) instead.')

        self.discounts = discounts
        tunedDLM._setDiscounts(discounts, change_component=True)
        return tunedDLM

    def getDiscounts(self):
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
        if a < 0.1:
            return 0.1

        if a >= 1:
            return 0.99999

        return a
