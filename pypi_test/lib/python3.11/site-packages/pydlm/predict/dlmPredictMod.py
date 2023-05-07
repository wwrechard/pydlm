from copy import deepcopy
from numpy import matrix
from pydlm.predict._dlmPredict import _dlmPredict

class dlmPredictModule(_dlmPredict):
    """ A dlm module containing all prediction methods.
    """


    # One day ahead prediction function
    def predict(self, date=None, featureDict=None):
        """ One day ahead predict based on the current data.

        The predict result is based on all the data before date and predict the
        observation at date + days. 

        The prediction could be on the last day and into the future or in 
        the middle of the time series and ignore the rest. For predicting into
        the future, the new features must be supplied to featureDict. For 
        prediction in the middle, the user can still supply the features which
        will be used priorily. The old features will be used if featureDict is
        None.

        Args:
            date: the index when the prediction based on. Default to the
                  last day.
            featureDict: the feature set for the dynamic Components, in a form
                  of {"component_name": feature}. If the featureDict is not
                  supplied, then the algo reuse those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.

        Returns:
            A tuple. (Predicted observation, variance of the predicted
            observation)

        """
        # the default prediction date
        if date is None:
            date = self.n - 1

        # check if the data on the date has been filtered
        if date > self.result.filteredSteps[1]:
            raise NameError('Prediction can only be made right' +
                            ' after the filtered date')

        # Clear the existing predictModel before the deepcopy to avoid recurrent
        # recurrent copy which could explode the memory and complexity.
        self._predictModel = None
        self._predictModel = deepcopy(self)
        return self._predictModel._oneDayAheadPredict(date=date,
                                                      featureDict=featureDict)


    def continuePredict(self, featureDict=None):
        """ Continue prediction after the one-day ahead predict.

        If users want to have a multiple day prediction, they can opt to use
        continuePredict after predict with new features contained in
        featureDict. For example,

        >>> # predict 3 days after the last day
        >>> myDLM.predict(featureDict=featureDict_day1)
        >>> myDLM.continuePredict(featureDict=featureDict_day2)
        >>> myDLM.continuePredict(featureDict=featureDict_day3)

        The featureDict acts the same way as in predict().

        Args:
            featureDict: the feature set for the dynamic components, stored
                         in a for of {"component name": vector}. If the set
                         was not supplied, then the algo will re-use the old
                         feature. For days beyond the data, the featureDict
                         for every dynamic component must be provided.

        Returns:
            A tupe. (predicted observation, variance)
        """
        if self._predictModel is None:
            raise NameError('continuePredict has to come after predict.')

        return self._predictModel._continuePredict(featureDict=featureDict)


    # N day ahead prediction
    def predictN(self, N=1, date=None, featureDict=None):
        """ N day ahead prediction based on the current data.

        This function is a convenient wrapper of predict() and
        continuePredict(). If the prediction is into the future, i.e, > n, 
        the featureDict has to contain all feature vectors for multiple days
        for each dynamic component. For example, assume myDLM has a component
        named 'spy' which posseses two dimensions,

        >>> featureDict_3day = {'spy': [[1, 2],[2, 3],[3, 4]]}
        >>> myDLM.predictN(N=3, featureDict=featureDict_3day)

        Args:
            N:    The length of days to predict.
            date: The index when the prediction based on. Default to the
                  last day.
            FeatureDict: The feature set for the dynamic Components, in a form
                  of {"component_name": feature}, where the feature must have
                  N elements of feature vectors. If the featureDict is not
                  supplied, then the algo reuse those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.

        Returns:
            A tuple of two lists. (Predicted observation, variance of the predicted
            observation)

        """
        if N < 1:
            raise NameError('N has to be greater or equal to 1')
        # Take care if features are numpy matrix
        if featureDict is not None:
            for name in featureDict:
                if isinstance(featureDict[name], matrix):
                    featureDict[name] = featureDict[name].tolist()
        predictedObs = []
        predictedVar = []

        # call predict for the first day
        getSingleDayFeature = lambda f, i: ({k: v[i] for k, v in f.items()}
                                            if f is not None else None)
        # Construct the single day featureDict
        featureDictOneDay = getSingleDayFeature(featureDict, 0)
        (obs, var) = self.predict(date=date, featureDict=featureDictOneDay)
        predictedObs.append(obs)
        predictedVar.append(var)

        # Continue predicting the remaining days
        for i in range(1, N):
            featureDictOneDay = getSingleDayFeature(featureDict, i)
            (obs, var) = self.continuePredict(featureDict=featureDictOneDay)
            predictedObs.append(obs)
            predictedVar.append(var)
        return (self._1DmatrixToArray(predictedObs),
                self._1DmatrixToArray(predictedVar))
