# This is the major class for fitting time series data using the
# dynamic linear model. dlm is a subclass of builder, with adding the
# Kalman filter functionality for filtering the data

from pydlm.modeler.builder import builder
from pydlm.modeler.component import component
from pydlm.base.kalmanFilter import kalmanFilter

class dlm:

    # define the basic members
    # initialize the result
    def __init__(self, data):

        self.data = data
        self.n = len(data)
        self.result = self.__result__(self.n)
        self.builder = builder()
        self.Filter = None

    # add component
    def add(self, component):
        self.builder.add(component)

    # list all components
    def ls(self):
        self.builder.ls()

    # delete one component
    def delete(self, index):
        self.builder.delete(index)

    # initialize the builder
    def __initialize__(self):
        self.builder.initialize()
        self.Filter = kalmanFilter(discount = self.builder.discount)
        
    # use the forward filter to filter the data
    # start: the place where the filter started
    # end: the place where the filter ended
    # save: the index for dates where the filtered results should be saved,
    #       could be 'all' or 'end'
    # isForget: indicate where the filter should use the previous state as prior
    #         or just use the prior information from builder
    def __forwardFilter__(self, start = 0, end = None, save = 'all', isForget = False):
        # the default value for end
        if end is None:
            end = self.n - 1

        # also we need to make we save consectively
        if save == 'all' and start > self.__result__.filteredSteps[1] + 1:
            raise NameError('The data before start date has yet to be filtered!')
        elif save == 'end' and end > self.__result__.filteredSteps[1] + 1:
            raise NameError('The data before the saved date has yet to be filtered!')
        
        # first we need to initialize the model to the correct status
        # if the start point is 0 or we want to forget the previous result
        if start == 0 or isForget:
            self.builder.model.state = self.builder.statePrior
            self.builder.model.sysVar = self.builder.sysVarPrior
            self.builder.model.df = 0

        # otherwise we use the result on the previous day as the prior
        else:
            self.builder.model.state = self.__result__.filteredState[start - 1]
            self.builder.model.sysVar = self.__result__.filteredCov[start - 1]
            self.builder.model.df = start
        
        # we run the forward filter sequentially
        for step in range(start, end + 1):
            
            # first check whether we need to update evaluation or not
            if len(self.builder.dynamicComponents) > 0:
                self.builder.updateEvaluation(step)

            # then we use the updated model to filter the state    
            self.Filter.forwardFilter(self.builder.model, self.data[step])

            # extract the result and record
            if save == 'all':
                self.__result__.filteredObs[step] = self.builder.model.obs
                self.__result__.predictedObs[step] = self.builder.model.prediction.obs
                self.__result__.filteredObsVar[step] = self.builder.model.obsVar
                self.__result__.predictedObsVar[step] = self.builder.model.prediction.obsVar     
                self.__result__.filteredState[step] = self.builder.model.state
                self.__result__.predictedState[step] = self.builder.model.prediction.state
                self.__result__.filteredCov[step] = self.builder.model.sysVar
                self.__result__.predictedCov[step] = self.builder.model.prediction.sysVar
                self.__result__.noiseVar[step] = self.builder.model.noiseVar
                
        # if we just need to save the last step
        self.__result__.filteredObs[end] = self.builder.model.obs
        self.__result__.predictedObs[end] = self.builder.model.prediction.obs
        self.__result__.filteredObsVar[end] = self.builder.model.obsVar
        self.__result__.predictedObsVar[end] = self.builder.model.prediction.obsVar
        self.__result__.filteredState[end] = self.builder.model.state
        self.__result__.predictedState[end] = self.builder.model.prediction.state
        self.__result__.filteredCov[end] = self.builder.model.sysVar
        self.__result__.predictedCov[end] = self.builder.model.prediction.sysVar
        self.__result__.noiseVar[end] = self.builder.model.noiseVar

        self.__result__.filteredSteps = (0, end)

    # use the backward smooth to smooth the state
    # start: the last date of the backward filtering chain
    # days: number of days to go back from start 
    def __backwardSmoother__(self, start = None, days = None):
        # the default start date is the most recent date
        if start is None:
            start = self.n - 1

        # the default backward days number is the total length
        if days is None:
            end = 0
        else:
            end = max(start - days, 0)

        # the forwardFilter has to be run before the smoother
        if self.__result__.filteredSteps[1] < start:
            raise NameError('The last day has to be filtered before smoothing!')
        else:
            # and we record the most recent day which does not need to be smooth
            self.__result__.smoothedState[start] = self.__result__.filteredState[start]
            self.__result__.smoothedObs[start] = self.__result__.filteredObs[start]
            self.__result__.smoothedCov[start] = self.__result__.filteredCov[start]
            self.__result__.smoothedObsVar[start] = self.__result__.filteredObsVar[start]
            self.builder.model.noiseVar = self.__result__.noiseVar[start]
            self.builder.model.state = self.__result__.smoothedState[start]
            self.builder.model.sysVar = self.__result__.smoothedCov[start]

        # we smooth the result sequantially from start - 1 to end
        dates = range(end, start)
        dates.reverse()
        for day in dates:
            # we first update the model to be correct status before smooth
            self.builder.model.prediction.state = self.__result__.predictedState[day + 1]
            self.builder.model.prediction.sysVar = self.__result__.predictedCov[day + 1]

            if len(self.builder.dynamicComponents) > 0:
                self.builder.updateEvaluation(day)

            # then we use the backward filter to filter the result
            self.Filter.backwardSmoother(model = self.builder.model, \
                                         rawState = self.__result__.filteredState[day], \
                                         rawSysVar = self.__result__.filteredCov[day])

            # extract the result
            self.__result__.smoothedState[day] = self.builder.model.state
            self.__result__.smoothedObs[day] = self.builder.model.obs
            self.__result__.smoothedCov[day] = self.builder.model.sysVar
            self.__result__.smoothedObsVar[day] = self.builder.model.obsVar
            
    # an inner class to store all results
    class __result__:
        # quantites to record the result
        def __init__(self, n):
            self.filteredObs = [None] * n
            self.predictedObs = [None] * n
            self.smoothedObs = [None] * n
            self.filteredObsVar = [None] * n
            self.predictedObsVar = [None] * n
            self.smoothedObsVar = [None] * n
            self.noiseVar = [None] * n
            
            self.filteredState = [None] * n
            self.predictedState = [None] * n
            self.smoothedState = [None] * n
            
            self.filteredCov = [None] * n
            self.predictedCov = [None] * n
            self.smoothedCov = [None] * n
            
            # quantity to indicate the current status
            self.filteredSteps = (0, -1)
            self.smoothedSteps = (0, -1)
    
