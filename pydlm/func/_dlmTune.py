"""
===============================================================================

The code for all tuning methods

===============================================================================

"""

from pydlm.func._dlm import _dlm


class _dlmTune(_dlm):
    """ The main class containing all tuning methods.

    Methods:
        _getMSE: obtain the fitting model one-day ahead prediction MSE.
        _getDiscounts: obtain the discounts (for different components).
        _setDiscounts: set discounts for different components.
    """

    # get the mse from the model
    def _getMSE(self):
        
        if not self.initialized:
            raise NameError('need to fit the model first')

        if self.result.filteredSteps[1] == -1:
            raise NameError('need to run forward filter first')

        mse = 0
        for i in range(self.result.filteredSteps[0],
                       self.result.filteredSteps[1] + 1):
            if self.data[i] is not None:
                mse += (self.data[i] - self.result.predictedObs[i]) ** 2

        mse = mse / (self.result.filteredSteps[1] + 1 -
                      self.result.filteredSteps[0])
        return mse[0,0]

    # get the discount from the model
    def _getDiscounts(self):
        
        if not self.initialized:
            raise NameError('need to fit the model before one can' +
                            'fetch the discount factors')

        discounts = []
        for comp in self.builder.componentIndex:
            indx = self.builder.componentIndex[comp]
            discounts.append(self.builder.discount[indx[0]])
        return discounts

    # set the model discount, this should never expose to the user
    # change the discount in the component would change the whole model.
    # change those in filter and builder only change the discounts
    # temporarily and will be corrected if we rebuild the model.
    def _setDiscounts(self, discounts, change_component=False):

        if not self.initialized:
            raise NameError('need to fit the model first')

        for i, comp in enumerate(self.builder.componentIndex):
            indx = self.builder.componentIndex[comp]
            self.builder.discount[indx[0]: (indx[1] + 1)] = discounts[i]
            if change_component:
                component = self._fetchComponent(name=comp)
                component.discount = self.builder.discount[indx[0]: (indx[1] + 1)]

        self.Filter.updateDiscount(self.builder.discount)
        self.result.filteredSteps = [0, -1]

