from pydlm.core._dlm import _dlm

class dlmPlotModule(_dlm):
    """ A dlm module containing all plot methods
    """
    def __init__(self, data, **options):
        super(dlmPlotModule, self).__init__(data, **options)

        # indicate whether the plot modules has been loaded.
        # We add this flag, since we only import plot module
        # when they are called to avoid any error due to
        # plot that blocks using this package. This flag can
        # help without doing import-check (expensive) every time
        # when plot function is called.
        self.plotLibLoaded = False


    def turnOn(self, switch):
        """ "turn on" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch on. \n
                    'filtered plot', 'filter' to plot filtered results\n
                    'smoothed plot', 'smooth' to plot smoothed results\n
                    'predict plot', 'predict', to plot one-step ahead results\n
                    'confidence interval', 'confierence', 'CI' to plot CI's\n
                    'data points', 'data', 'data point' to plot original data\n
                    'multiple', 'separate' to plot results in separate
                    figures\n
                    'fitted dots', 'fitted' to plot fitted results with dots
        """
        if switch in set(['filtered plot', 'filter',
                          'filtered results', 'filtering']):
            self.options.plotFilteredData = True
        elif switch in set(['smoothed plot', 'smooth',
                            'smoothed results', 'smoothing']):
            self.options.plotSmoothedData = True
        elif switch in set(['predict plot', 'predict',
                            'predicted results', 'prediction']):
            self.options.plotPredictedData = True
        elif switch in set(['confidence interval', 'confidence',
                            'interval', 'CI', 'ci']):
            self.options.showConfidenceInterval = True
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = True
        elif switch in set(['multiple', 'multiple plots',
                            'separate plots', 'separate']):
            self.options.separatePlot = True
        elif switch in set(['fitted dots', 'fitted results',
                            'fitted data', 'fitted']):
            self.options.showFittedPoint = True
        else:
            raise NameError('no such options')


    def turnOff(self, switch):
        """ "turn off" Operation for the dlm plotting options.

        Args:
            switch: The key word to switch off. \n
                    'filtered plot', 'filter' to not plot filtered results\n
                    'smoothed plot', 'smooth' to not plot smoothed results\n
                    'predict plot', 'predict', to not plot one-step ahead
                    results\n
                    'confidence interval', 'confierence', 'CI'
                    to not plot CI's\n
                    'data points', 'data', 'data point' to not
                    plot original data\n
                    'multiple', 'separate' to not plot results
                    in separate figures\n
                    'fitted dots', 'fitted' to not plot fitted
                    results with dots

        """
        if switch in set(['filtered plot', 'filter', 'filtered results',
                          'filtering']):
            self.options.plotFilteredData = False
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results',
                            'smoothing']):
            self.options.plotSmoothedData = False
        elif switch in set(['predict plot', 'predict', 'predicted results',
                            'prediction']):
            self.options.plotPredictedData = False
        elif switch in set(['confidence interval', 'confidence', 'interval',
                            'CI', 'ci']):
            self.options.showConfidenceInterval = False
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.showDataPoint = False
        elif switch in set(['multiple', 'multiple plots', 'separate plots',
                            'separate']):
            self.options.separatePlot = False
        elif switch in set(['fitted dots', 'fitted results', 'fitted data',
                            'fitted']):
            self.options.showFittedPoint = False
        else:
            raise NameError('no such options')


    def setColor(self, switch, color):
        """ "set" Operation for the dlm plotting colors

        Args:
            switch: key word. Controls over
                    filtered/smoothed/predicted results,
            color: the color for the corresponding keyword.
        """
        if switch in set(['filtered plot', 'filter', 'filtered results',
                          'filtering']):
            self.options.filteredColor = color
        elif switch in set(['smoothed plot', 'smooth', 'smoothed results',
                            'smoothing']):
            self.options.smoothedColor = color
        elif switch in set(['predict plot', 'predict', 'predicted results',
                            'prediction']):
            self.options.predictedColor = color
        elif switch in set(['data points', 'data point', 'points', 'data']):
            self.options.dataColor = color
        else:
            raise NameError('no such options')


    def setConfidence(self, p=0.95):
        """ Set the confidence interval for the plot

        """
        assert p >= 0 and p <= 1
        self.options.confidence = p


    def setIntervalType(self, intervalType):
        """ Set the confidence interval type

        """
        if intervalType == 'ribbon' or intervalType == 'line':
            self.options.intervalType = intervalType
        else:
            raise NameError('No such type for confidence interval.')


    def resetPlotOptions(self):
        """ Reset the plotting option for the dlm class

        """
        self.options.plotOriginalData = True
        self.options.plotFilteredData = True
        self.options.plotSmoothedData = True
        self.options.plotPredictedData = True
        self.options.showDataPoint = False
        self.options.showFittedPoint = False
        self.options.showConfidenceInterval = True
        self.options.dataColor = 'black'
        self.options.filteredColor = 'blue'
        self.options.predictedColor = 'green'
        self.options.smoothedColor = 'red'
        self.options.separatePlot = True
        self.options.confidence = 0.95
        self.options.intervalType = 'ribbon'


    # plot the result according to the options
    def plot(self, name='main'):
        """ The main plot function. The dlmPlot and the matplotlib will only be loaded
        when necessary.

        Args:
            name: component to plot. Default to 'main', in which we plot the
                  filtered time series. If a component name is given
                  It plots the mean of the component, i.e., the observed value
                  that attributes to that particular component, which equals to
                  evaluation * latent states for that particular component.

        """

        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # initialize the figure
        dlmPlot.plotInitialize()

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        # plot the main time series after filtering
        if name == 'main':
            # if we just need one plot
            if self.options.separatePlot is not True:
                dlmPlot.plotInOneFigure(time=time,
                                        data=self.data,
                                        result=self.result,
                                        options=self.options)
            # otherwise, we plot in multiple figures
            else:
                dlmPlot.plotInMultipleFigure(time=time,
                                             data=self.data,
                                             result=self.result,
                                             options=self.options)

        # plot the component after filtering
        elif self._checkComponent(name):
            # create the data for ploting
            data = {}
            if self.options.plotFilteredData:
                data['filteredMean'] = self.getMean(
                    filterType='forwardFilter', name=name)
                data['filteredVar'] = self.getVar(
                    filterType='forwardFilter', name=name)

            if self.options.plotSmoothedData:
                data['smoothedMean'] = self.getMean(
                    filterType='backwardSmoother', name=name)
                data['smoothedVar'] = self.getVar(
                    filterType='backwardSmoother', name=name)

            if self.options.plotPredictedData:
                data['predictedMean'] = self.getMean(
                    filterType='predict', name=name)
                data['predictedVar'] = self.getVar(
                    filterType='predict', name=name)

            if len(data) == 0:
                raise NameError('Nothing is going to be drawn, due to ' +
                                'user choices.')
            data['name'] = name
            dlmPlot.plotComponent(time=time,
                                  data=data,
                                  result=self.result,
                                  options=self.options)
        dlmPlot.plotout()


    def plotCoef(self, name, dimensions=None):
        """ Plot function for the latent states (coefficents of dynamic
        component).

        Args:
            name: the name of the component to plot.
                  It plots the latent states for the component. If dimension of
                  the given component is too high, we truncate
                  to the first five. Or the user can supply the ideal
                  dimensions for plot in the dimensions parameter.
            dimensions: dimensions will be used
                        as the indexes to plot within that component latent
                        states.
        """
        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        # provide a fake time for plotting
        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        # plot the latent states for a given component
        if self._checkComponent(name):

            # find its coordinates in the latent state
            indx = self.builder.componentIndex[name]
            # get the real latent states
            coordinates = range(indx[0], (indx[1] + 1))
            # if user supplies the dimensions
            if dimensions is not None:
                coordinates = [coordinates[i] for i in dimensions]

            # otherwise, if there are too many latent states, we
            # truncated to the first five.
            elif len(coordinates) > 5:
                coordinates = coordinates[:5]

            dlmPlot.plotLatentState(time=time,
                                    coordinates=coordinates,
                                    result=self.result,
                                    options=self.options,
                                    name=name)
        else:
            raise NameError('No such component.')

        dlmPlot.plotout()


    def plotPredictN(self, N=1, date=None, featureDict=None):
        """
        Function to plot the N-day prediction results.

        The input is the same as `dlm.predictN`. For details,
        please refer to that function.

        Args:
            N:    The length of days to predict.
            date: The index when the prediction based on. Default to the
                  last day.
            FeatureDict: The feature set for the dynamic Components, in a form
                  of {"component_name": feature}, where the feature must have
                  N elements of feature vectors. If the featureDict is not
                  supplied, then the algo reuses those stored in the dynamic
                  components. For dates beyond the last day, featureDict must
                  be supplied.
        """
        if date is None:
            date = self.n - 1

        # load the library only when needed
        # import pydlm.plot.dlmPlot as dlmPlot
        self.loadPlotLibrary()

        # provide a fake time for plotting
        if self.time is None:
            time = range(len(self.data))
        else:
            time = self.time

        # change option setting if some results are not available
        if not self.initialized:
            raise NameError('The model must be constructed and' +
                            ' fitted before ploting.')

        # check the filter status and automatically turn off bad plots
        self._checkPlotOptions()

        predictedTimeRange = range(date, date + N)
        predictedData, predictedVar = self.predictN(
            N=N, date=date, featureDict=featureDict)
        dlmPlot.plotPrediction(
            time=time, data=self.data,
            predictedTime=predictedTimeRange,
            predictedData=predictedData,
            predictedVar=predictedVar,
            options=self.options)

        dlmPlot.plotout()



    def loadPlotLibrary(self):
        if not self.plotLibLoaded:
            global dlmPlot
            import pydlm.plot.dlmPlot as dlmPlot
            self.plotLibLoaded = True
