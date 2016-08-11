from pydlm.base.kalmanFilter import kalmanFilter
from pydlm.modeler.builder import builder

# this class defines the basic functionalities for dlm, which is not supposed to
# be used by the user. Most functionality in the main dlm will be constructed
# by using the hidden functions in this class
class _dlm:
    # define the basic members
    # initialize the result
    def __init__(self, data):

        self.data = data
        self.n = len(data)
        self.result = None
        self.builder = builder()
        self.Filter = None
        self.initialized = False
    
    # initialize the builder
    def _initialize(self):
        self.builder.initialize()
        self.Filter = kalmanFilter(discount = self.builder.discount)
        self.result = self._result(self.n)
        self.initialized = True
        
    # use the forward filter to filter the data
    # start: the place where the filter started
    # end: the place where the filter ended
    # save: the index for dates where the filtered results should be saved,
    #       could be 'all' or 'end'
    # isForget: indicate where the filter should use the previous state as prior
    #         or just use the prior information from builder
    def _forwardFilter(self, start = 0, end = None, save = 'all', ForgetPrevious = False):
        
        # the default value for end
        if end is None:
            end = self.n - 1

        # to see if the ff need to run or not
        if start > end:
            return None
        
        # also we need to make we save consectively
#        if save == 'all' and start > self.result.filteredSteps[1] + 1:
#            raise NameError('The data before start date has yet to be filtered!')

        # for rolling window run, we need to make sure the saved date is consecutive
#        if save != 'all' and end > self.result.filteredSteps[1] + 1:
#            raise NameError('The previous date needs to be filtered for rolling window!')
        
        # first we need to initialize the model to the correct status
        # if the start point is 0 or we want to forget the previous result
        if start == 0 or ForgetPrevious:
            self._resetModelStatus()

        # otherwise we use the result on the previous day as the prior
        else:
            if start > self.result.filteredSteps[1] + 1:
                raise NameError('The data before start date \
                has yet to be filtered! Otherwise set ForgetPrevious to be True.\
                check the <filteredSteps> in <result> object.')
            self._setModelStatus(date = start - 1)
        
        # we run the forward filter sequentially
        for step in range(start, end + 1):
            
            # first check whether we need to update evaluation or not
            if len(self.builder.dynamicComponents) > 0:
                self.builder.updateEvaluation(step)

            # then we use the updated model to filter the state    
            self.Filter.forwardFilter(self.builder.model, self.data[step])

            # extract the result and record
            if save == 'all' or save == step:
                self._copy(model = self.builder.model, \
                              result = self.result, \
                              step = step, \
                              filterType = 'forwardFilter')

#        self.result.filteredSteps = (0, end)


    # use the backward smooth to smooth the state
    # start: the last date of the backward filtering chain
    # days: number of days to go back from start 
    def _backwardSmoother(self, start = None, days = None):
        # the default start date is the most recent date
        if start is None:
            start = self.n - 1
        
        # the default backward days number is the total length
        if days is None:
            end = 0
        else:
            end = max(start - days, 0)
        
        # the forwardFilter has to be run before the smoother
        if self.result.filteredSteps[1] < start:
            raise NameError('The last day has to be filtered before smoothing! \
            check the <filteredSteps> in <result> object.')

        # empty smoothing chain, return None
        if start <= end:
            return None
        
        # and we record the most recent day which does not need to be smooth
        self.result.smoothedState[start] = self.result.filteredState[start]
        self.result.smoothedObs[start] = self.result.filteredObs[start]
        self.result.smoothedCov[start] = self.result.filteredCov[start]
        self.result.smoothedObsVar[start] = self.result.filteredObsVar[start]
        
        self.builder.model.noiseVar = self.result.noiseVar[start]
        self.builder.model.state = self.result.smoothedState[start]
        self.builder.model.sysVar = self.result.smoothedCov[start]

        # we smooth the result sequantially from start - 1 to end
        dates = range(end, start)
        dates.reverse()
        for day in dates:
            # we first update the model to be correct status before smooth
            self.builder.model.prediction.state = self.result.predictedState[day + 1]
            self.builder.model.prediction.sysVar = self.result.predictedCov[day + 1]

            if len(self.builder.dynamicComponents) > 0:
                self.builder.updateEvaluation(day)

            # then we use the backward filter to filter the result
            self.Filter.backwardSmoother(model = self.builder.model, \
                                         rawState = self.result.filteredState[day], \
                                         rawSysVar = self.result.filteredCov[day])

            # extract the result
            self._copy(model = self.builder.model, \
                          result = self.result, \
                          step = day, \
                          filterType = 'backwardSmoother')
            
#        self.result.smoothedSteps = (end, start)

    # Forecast the result based on filtered chains
    def _predict(self, date = None, days = 1):
        if date is None:
            date = self.n - 1

        # reset the date to the date we are interested in
        self._setModelStatus(date = date)
        self.builder.model.prediction.step = 0
        for i in range(days):
            self.Filter.predict(self.builder.model)
            
    # to set model to a specific date
    def _setModelStatus(self, date = 0):
        if date < self.result.filteredSteps[0] or date > self.result.filteredSteps[1]:
            raise NameError('The date has yet to be filtered yet.\
            check the <filteredSteps> in <result> object.')
        
        self._reverseCopy(model = self.builder.model, \
                          result = self.result, \
                          step = date)
        
    # reset model to initial status
    def _resetModelStatus(self):
        self.builder.model.state = self.builder.statePrior
        self.builder.model.sysVar = self.builder.sysVarPrior
        self.builder.model.noiseVar = self.builder.noiseVar
        self.builder.model.df = 1
        self.builder.model.initializeObservation()
        
    # an inner class to store all results
    class _result:
        # quantites to record the result
        def __init__(self, n):
            self.filteredObs = [None] * n
            self.predictedObs = [None] * n
            self.smoothedObs = [None] * n
            self.filteredObsVar = [None] * n
            self.predictedObsVar = [None] * n
            self.smoothedObsVar = [None] * n
            self.noiseVar = [None] * n
            self.df = [None] * n
            
            self.filteredState = [None] * n
            self.predictedState = [None] * n
            self.smoothedState = [None] * n
            
            self.filteredCov = [None] * n
            self.predictedCov = [None] * n
            self.smoothedCov = [None] * n
            
            # quantity to indicate the current status
            self.filteredSteps = (0, -1)
            self.smoothedSteps = (0, -1)
    

    # a function used to copy result from the model to the result
    def _copy(self, model, result, step, filterType):
        if filterType == 'forwardFilter':
            result.filteredObs[step] = model.obs
            result.predictedObs[step] = model.prediction.obs
            result.filteredObsVar[step] = model.obsVar
            result.predictedObsVar[step] = model.prediction.obsVar
            result.filteredState[step] = model.state
            result.predictedState[step] = model.prediction.state
            result.filteredCov[step] = model.sysVar
            result.predictedCov[step] = model.prediction.sysVar
            result.noiseVar[step] = model.noiseVar
            result.df[step] = model.df
            
        elif filterType == 'backwardSmoother':
            result.smoothedState[step] = model.state
            result.smoothedObs[step] = model.obs
            result.smoothedCov[step] = model.sysVar
            result.smoothedObsVar[step] = model.obsVar

    def _reverseCopy(self, model, result, step):
        model.obs = result.filteredObs[step]
        model.prediction.obs = result.predictedObs[step]
        model.obsVar = result.filteredObsVar[step] 
        model.prediction.obsVar = result.predictedObsVar[step]
        model.state = result.filteredState[step]
        model.prediction.state = result.predictedState[step]
        model.sysVar = result.filteredCov[step]
        model.prediction.sysVar = result.predictedCov[step]
        model.noiseVar = result.noiseVar[step]
        model.df = result.df[step]
