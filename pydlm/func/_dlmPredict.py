"""
===============================================================================

The code for all predicting methods

===============================================================================

"""

from pydlm.func._dlm import _dlm


class _dlmPredict(_dlm):
    """ The main class containing all prediction methods.

    Methods:
        _oneDayAheadPredict: predict one day a head.
        _continuePredict: continue predicting one day after _oneDayAheadPredict
    """

    # Note the following functions will modify the status of the model, so they
    # shall not be directly call through the main model if this behavior is not
    # desired.
    
    # featureDict contains all the features for prediction.
    # It is a dictionary with key equals to the name of the component and
    # the value as the new feature (a list). The function
    # will first use the features provided in this feature dict, if not
    # found, it will fetch the default feature from the component. If
    # it could not find feature for some component, it returns an error
    # The intermediate result will be stored in result.predictStatus as
    # (start_date, next_pred_date, [all_predicted_values]), which will be
    # used by _continuePredict.
    def _oneDayAheadPredict(self, date, featureDict=None):
        """ One day ahead prediction based on the date and the featureDict.
        The prediction could be on the last day and into the future or in 
        the middle of the time series and ignore the rest. For predicting into
        the future, the new features must be supplied to featureDict. For 
        prediction in the middle, the user can still supply the features which
        will be used priorily. The old features will be used if featureDict is
        None.

        Args:
            date: the prediction starts (based on the observation before and
                  on this date)
            featureDict: the new feature value for some dynamic components.
                         must be specified in a form of {component_name: value}
                         if the feature for some dynamic component is not
                         supplied. The algorithm will use the features from
                         the old data. (which means if the prediction is out
                         of sample, then all dynamic component must be provided
                         with the new feature value)

        Returns:
            A tuple of (predicted_mean, predicted_variance)
        """
        if date > self.n - 1:
            raise NameError('The date is beyond the data range.')

        # get the correct status of the model
        self._setModelStatus(date=date)
        self._constructEvaluationForPrediction(
            date=date + 1,
            featureDict=featureDict,
            padded_data=self.padded_data[:(date + 1)])
        
        # initialize the prediction status
        self.builder.model.prediction.step = 0

        # start predicting
        self.Filter.predict(self.builder.model)

        predictedObs = self.builder.model.prediction.obs
        predictedObsVar = self.builder.model.prediction.obsVar
        self.result.predictStatus = [
            date,                   # start_date
            date + 1,               # current_date
            [predictedObs[0, 0]]    # all historical predictions
        ]

        return (predictedObs, predictedObsVar)

    def _continuePredict(self, featureDict=None):
        """ Continue predicting one day after _oneDayAheadPredict or
        after _continuePredict. After using
        _oneDayAheadPredict, the user can continue predicting by using
        _continuePredict. The featureDict act the same as in
        _oneDayAheadPredict.

        Args:
            featureDict: the new feature value for some dynamic components.
                         see @_oneDayAheadPredict

        Returns:
            A tuple of (predicted_mean, predicted_variance)
        """
        if self.result.predictStatus is None:
            raise NameError('_continoousPredict can only be used after ' +
                            '_oneDayAheadPredict')
        startDate = self.result.predictStatus[0]
        currentDate = self.result.predictStatus[1]

        self._constructEvaluationForPrediction(
            date=currentDate + 1,
            featureDict=featureDict,
            padded_data=self.padded_data[:(startDate + 1)] +
                        self.result.predictStatus[2])
        self.Filter.predict(self.builder.model)
        predictedObs = self.builder.model.prediction.obs
        predictedObsVar = self.builder.model.prediction.obsVar
        self.result.predictStatus[1] += 1
        self.result.predictStatus[2].append(predictedObs[0, 0])
        return (predictedObs, predictedObsVar)

    # This function will modify the status of the object, use with caution.
    def _constructEvaluationForPrediction(self,
                                          date,
                                          featureDict=None,
                                          padded_data=None):
        """ Construct the evaluation matrix based on date and featureDict.

        Used for prediction. Features provided in the featureDict will be used
        preferrably. If the feature is not found in featureDict, the algorithm
        will seek it based on the old data and the date.

        Args:
            featureDict: a dictionary containing {dynamic_component_name: value}
                         for update the feature for the corresponding component.
            date: if a dynamic component name is not found in featureDict, the
                  algorithm is using its old feature on the given date.
            padded_data: is the mix of the raw data and the predicted data. It is
                  used by auto regressor.

        """
        # New features are provided. Update dynamic componnet.
        # We distribute the featureDict back to dynamicComponents. If the date is
        # out of bound, we append the feature to the feature set. If the date is
        # within range, we replace the old feature with the new feature.
        if featureDict is not None:
            for name in featureDict:
                if name in self.builder.dynamicComponents:
                    comp = self.builder.dynamicComponents[name]                    
                    # the date is within range
                    if date < comp.n:
                        comp.features[date] = featureDict[name]
                        comp.n += 1
                    elif date < comp.n + 1:
                        comp.features.append(featureDict[name])
                        comp.n += 1
                    else:
                        raise NameError("Feature is missing between the last predicted " +
                                        "day and the new day")
                        
        self.builder.updateEvaluation(date, padded_data)
