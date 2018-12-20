from copy import deepcopy
from pydlm.base.tools import getInterval
from pydlm.access._dlmGet import _dlmGet


class dlmAccessModule(_dlmGet):
    """ A dlm module for all the access methods
    """


    def getAll(self):
        """ get all the _result class which contains all results

        Returns:
            The @result object containing all computed results.

        """
        return deepcopy(self.result)


    def getMean(self, filterType='forwardFilter', name='main'):
        """ get mean for data or component.

        If the working dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of mean to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get mean. When name = 'main', then it
                  returns the filtered mean for the time series. When
                  name = some component's name, then it returns the filtered
                  mean for that component. Default to 'main'.

        Returns:
            A list of the time series observations based on the choice

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1 # To get the result for the last date.
        # get the mean for the fitlered data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                return self._1DmatrixToArray(
                    self.result.filteredObs[start:end])
            elif filterType == 'backwardSmoother':
                return self._1DmatrixToArray(
                    self.result.smoothedObs[start:end])
            elif filterType == 'predict':
                return self._1DmatrixToArray(
                    self.result.predictedObs[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the mean for the component
        self._checkComponent(name)
        return self._getComponentMean(name=name,
                                      filterType=filterType,
                                      start=start, end=(end - 1))


    def getVar(self, filterType='forwardFilter', name='main'):
        """ get the variance for data or component.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating
        the actual filtered dates.

        Args:
            filterType: the type of variance to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get variance. When name = 'main', then it
                  returns the filtered variance for the time series. When
                  name = some component's name, then it returns the filtered
                  variance for that component. Default to 'main'.

        Returns:
            A list of the filtered variances based on the choice.

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # get the variance for the time series data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                return self._1DmatrixToArray(
                    self.result.filteredObsVar[start:end])
            elif filterType == 'backwardSmoother':
                return self._1DmatrixToArray(
                    self.result.smoothedObsVar[start:end])
            elif filterType == 'predict':
                return self._1DmatrixToArray(
                    self.result.predictedObsVar[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the variance for the component
        self._checkComponent(name)
        return self._getComponentVar(name=name, filterType=filterType,
                                     start=start, end=(end - 1))


    def getResidual(self, filterType='forwardFilter'):
        """ get the residuals for data after filtering or smoothing.

        If the working dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of residuals to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.

        Returns:
            A list of residuals based on the choice

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1 # To get the result for the last date.
        # get the mean for the fitlered data
        # get out of the matrix form
        if filterType == 'forwardFilter':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.filteredObs[i]
                 for i in range(start, end)])
        elif filterType == 'backwardSmoother':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.smoothedObs[i]
                 for i in range(start, end)])
        elif filterType == 'predict':
            return self._1DmatrixToArray(
                [self.data[i] - self.result.predictedObs[i]
                 for i in range(start, end)])
        else:
            raise NameError('Incorrect filter type.')


    def getInterval(self, p=0.95, filterType='forwardFilter', name='main'):
        """ get the confidence interval for data or component.

        If the filtered dates are not
        (0, self.n - 1), then a warning will prompt stating the actual
        filtered dates.

        Args:
            p: The confidence level.
            filterType: the type of CI to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get CI. When name = 'main', then it
                  returns the confidence interval for the time series. When
                  name = some component's name, then it returns the confidence
                  interval for that component. Default to 'main'.

        Returns:
            A tuple with the first element being a list of upper bounds
            and the second being a list of the lower bounds.

        """
        # get the working date
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # get the mean and the variance for the time series data
        if name == 'main':
            # get out of the matrix form
            if filterType == 'forwardFilter':
                compMean = self._1DmatrixToArray(
                    self.result.filteredObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.filteredObsVar[start:end])
            elif filterType == 'backwardSmoother':
                compMean = self._1DmatrixToArray(
                    self.result.smoothedObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.smoothedObsVar[start:end])
            elif filterType == 'predict':
                compMean = self._1DmatrixToArray(
                    self.result.predictedObs[start:end])
                compVar = self._1DmatrixToArray(
                    self.result.predictedObsVar[start:end])
            else:
                raise NameError('Incorrect filter type.')

        # get the mean and variance for the component
        else:
            self._checkComponent(name)
            compMean = self._getComponentMean(name=name,
                                              filterType=filterType,
                                              start=start, end=(end - 1))
            compVar = self._getComponentVar(name=name,
                                            filterType=filterType,
                                            start=start, end=(end - 1))

        # get the upper and lower bound
        upper, lower = getInterval(compMean, compVar, p)
        return (upper, lower)


    def getLatentState(self, filterType='forwardFilter', name='all'):
        """ get the latent states for different components and filters.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of latent states to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get latent state. When name = 'all', then it
                  returns the latent states for the time series. When
                  name = some component's name, then it returns the latent
                  states for that component. Default to 'all'.

        Returns:
            A list of lists, standing for the latent states given
            the different choices.

        """
        # get the working dates
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # to return the full latent states
        if name == 'all':
            if filterType == 'forwardFilter':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.filteredState[start:end]))
            elif filterType == 'backwardSmoother':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.smoothedState[start:end]))
            elif filterType == 'predict':
                return list(map(lambda x: x if x is None
                                else self._1DmatrixToArray(x),
                                self.result.smoothedState[start:end]))
            else:
                raise NameError('Incorrect filter type.')

        # to return the latent state for a given component
        self._checkComponent(name)
        return list(map(lambda x: x if x is None else self._1DmatrixToArray(x),
                        self._getLatentState(name=name, filterType=filterType,
                                             start=start, end=(end - 1))))


    def getLatentCov(self, filterType='forwardFilter', name='all'):
        """ get the error covariance for different components and
        filters.

        If the filtered dates are not (0, self.n - 1),
        then a warning will prompt stating the actual filtered dates.

        Args:
            filterType: the type of latent covariance to be returned. Could be
                        'forwardFilter', 'backwardSmoother', and 'predict'.
                        Default to 'forwardFilter'.
            name: the component to get latent cov. When name = 'all', then it
                  returns the latent covariance for the time series. When
                  name = some component's name, then it returns the latent
                  covariance for that component. Default to 'all'.

        Returns:
            A list of numpy matrices, standing for the filtered latent
            covariance.

        """
        # get the working dates
        start, end = self._checkAndGetWorkingDates(filterType=filterType)
        end += 1
        # to return the full latent covariance
        if name == 'all':
            if filterType == 'forwardFilter':
                return self.result.filteredCov[start:end]
            elif filterType == 'backwardSmoother':
                return self.result.smoothedCov[start:end]
            elif filterType == 'predict':
                return self.result.smoothedCov[start:end]
            else:
                raise NameError('Incorrect filter type.')

        # to return the latent covariance for a given component
        self._checkComponent(name)
        return self._getLatentCov(name=name, filterType=filterType,
                                  start=start, end=(end - 1))
