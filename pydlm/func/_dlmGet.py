"""
===============================================================================

The code for all get methods

===============================================================================

"""
from numpy import dot
from pydlm.func._dlm import _dlm


class _dlmGet(_dlm):
    """ The class containing all get methods for dlm class.

    Methods:
        _getComponent: get the component if it is in dlm
        _getLatentState: get the latent state for a given component
        _getLatentCov: get the latent covariance for a given component
        _getComponentMean: get the mean of a given component
        _getComponentVar: get the variance of a given component
    """
    
    # function to get the corresponding latent state
    def _getLatentState(self, name, filterType, start, end):
        """ Get the latent states of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the latent states to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the latent states to be returned.
            end: the end date to be returned.

        Returns:
            A list of latent states.
        """
        end += 1
        indx = self.builder.componentIndex[name]
        patten = lambda x: x if x is None else x[indx[0]:(indx[1] + 1), 0]

        if filterType == 'forwardFilter':
            return list(map(patten, self.result.filteredState[start:end]))
        elif filterType == 'backwardSmoother':
            return list(map(patten, self.result.smoothedState[start:end]))
        elif filterType == 'predict':
            return list(map(patten, self.result.predictedState[start:end]))
        else:
            raise NameError('Incorrect filter type')

    # function to get the corresponding latent covariance
    def _getLatentCov(self, name, filterType, start, end):
        """ Get the latent covariance of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the latent covariance to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the latent covariance to be returned.
            end: the end date to be returned.

        Returns:
            A list of latent covariance.
        """
        end += 1
        indx = self.builder.componentIndex[name]
        patten = lambda x: x if x is None \
                 else x[indx[0]:(indx[1] + 1), indx[0]:(indx[1] + 1)]

        if filterType == 'forwardFilter':
            return list(map(patten, self.result.filteredCov[start:end]))
        elif filterType == 'backwardSmoother':
            return list(map(patten, self.result.smoothedCov[start:end]))
        elif filterType == 'predict':
            return list(map(patten, self.result.predictedCov[start:end]))
        else:
            raise NameError('Incorrect filter type')

    # function to get the component mean
    def _getComponentMean(self, name, filterType, start, end):
        """ Get the mean of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the mean to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the mean to be returned.
            end: the end date to be returned.

        Returns:
            A list of mean.
        """
        end += 1
        comp = self._fetchComponent(name)
        componentState = self._getLatentState(name=name,
                                              filterType=filterType,
                                              start=start, end=end)
        result = []
        for k, i in enumerate(range(start, end)):
            if name in self.builder.dynamicComponents:
                comp.updateEvaluation(i)
            elif name in self.builder.automaticComponents:
                comp.updateEvaluation(i, self.padded_data)
            result.append(dot(comp.evaluation,
                              componentState[k]).tolist()[0][0])
        return result

    # function to get the component variance
    def _getComponentVar(self, name, filterType, start, end):
        """ Get the variance of a given component.

        Args:
            name: the name of the component.
            filterType: the type of the variance to be returned.
                        could be "forwardFilter", "backwardSmoother" or
                        "predict".
            start: the start date for the variance to be returned.
            end: the end date to be returned.

        Returns:
            A list of variance.
        """
        end += 1
        comp = self._fetchComponent(name)
        componentCov = self._getLatentCov(name=name,
                                          filterType=filterType,
                                          start=start, end=end)
        result = []
        for k, i in enumerate(range(start, end)):
            if name in self.builder.dynamicComponents:
                comp.updateEvaluation(i)
            elif name in self.builder.automaticComponents:
                comp.updateEvaluation(i, self.padded_data)
            result.append(dot(
                dot(comp.evaluation,
                    componentCov[k]), comp.evaluation.T).tolist()[0][0])
        return result
