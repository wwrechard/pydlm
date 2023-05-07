from pydlm.tuner._dlmTune import _dlmTune
from pydlm.tuner.dlmTuner import modelTuner

class dlmTuneModule(_dlmTune):
    """ A dlm model containing all tuning methods
    """


    def getMSE(self):
        """ Get the one-day ahead prediction mean square error. The mse is
        estimated only for days that has been predicted.

        Returns:
            An numerical value
        """

        return self._getMSE()


    def tune(self, maxit=100):
        """ Automatic tuning of the discounting factors. 

        The method will call the model tuner class to use the default parameters
        to tune the discounting factors and change the discount factor permenantly.
        User needs to refit the model after tuning.
        
        If user wants a more refined tuning and not change any property of the
        existing model, they should opt to use the @modelTuner class.
        """
        simpleTuner = modelTuner()

        if self._printInfo:
            self.fitForwardFilter()
            print("The current mse is " + str(self.getMSE()) + '.')
        
        simpleTuner.tune(untunedDLM=self, maxit=maxit)
        self._setDiscounts(simpleTuner.getDiscounts(), change_component=True)
        
        if self._printInfo:
            self.fitForwardFilter()
            print("The new mse is " + str(self.getMSE()) + '.')
