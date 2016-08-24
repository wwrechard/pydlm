"""
==================================================================================

Code for plot dlm results

==================================================================================

This module provide the plotting functionality for dlm 

"""
import matplotlib.pyplot as plt
from pydlm.base.tools import getInterval


def plotInOneFigure(time, data, result, options):
    """
    Plot the dlm results in one figure
    
    Args:
        time: the time label
        data: the original data
        result: the fitted result from dlm class
        options: options for the plot, for details please refer to @dlm
    """
    # plot the original data
    plotData(time = time, data = data, showDataPoint = \
             options.showDataPoint, color = options.dataColor, label = 'time series')
    
    # plot fitered results if needed
    if options.plotFilteredData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time = time[start : end], \
                 data = to1dArray(result.filteredObs[start : end]), \
                 showDataPoint = options.showFittedPoint, \
                 color = options.filteredColor, \
                 label = 'filtered series')

        if options.showConfidenceInterval:
            upper, lower = getInterval(result.filteredObs[start : end], \
                                       result.filteredObsVar[start : end], \
                                       p = options.confidence)
                    
            plotInterval(time = time[start : end], \
                         upper = to1dArray(upper), lower = to1dArray(lower), \
                         color = options.filteredColor)
            
        # plot predicted results if needed
    if options.plotPredictedData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time = time[start : end], \
                 data = to1dArray(result.predictedObs), \
                 showDataPoint = options.showFittedPoint, \
                 color = options.predictedColor, \
                 label = 'one-day prediction')

        if options.showConfidenceInterval:
            upper, lower = getInterval(result.predictedObs[start : end], \
                                       result.filteredObsVar[start : end], \
                                       p = options.confidence)
            
            plotInterval(time = time[start:end], \
                         upper = to1dArray(upper), lower = to1dArray(lower), \
                         color = options.predictedColor)

    # plot smoothed results if needed
    if options.plotSmoothedData:
        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        plotData(time = time[start:end], \
                 data = to1dArray(result.smoothedObs), \
                 showDataPoint = options.showFittedPoint, \
                 color = options.smoothedColor, \
                 label = 'smoothed series')
            
        if options.showConfidenceInterval:
            upper, lower = getInterval(result.smoothedObs[start : end], \
                                       result.smoothedObsVar[start : end], \
                                       p = options.confidence)
            plotInterval(time = time[start:end], \
                         upper = to1dArray(upper), lower = to1dArray(lower), \
                         color = options.smoothedColor)
            
    plt.legend(loc='best', shadow = True) #, fontsize = 'x-large')   
    
def plotInMultipleFigure(time, data, result, options):
    """
    Plot the dlm results in multiple figure, each with one result and the original
    data
    
    Args:
        time: the time label
        data: the original data
        result: the fitted result from dlm class
        options: options for the plot, for details please refer to @dlm
    """
    # first compute how many plots are needed
    numOfPlots = options.plotFilteredData + options.plotPredictedData + \
                 options.plotSmoothedData
    size = (numOfPlots, 1)
    location = 1

    # plot all needed results   
    # plot fitered results if needed
    if options.plotFilteredData:
        # the location
        subplot(size, location)
        
        #plot original data
        plotData(time = time, data = data, showDataPoint = \
                 options.showDataPoint, color = options.dataColor, label = 'time series')
        
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        if start < end:
            plotData(time = time[start : end], \
                     data = to1dArray(result.filteredObs[start : end]), \
                     showDataPoint = options.showFittedPoint, \
                     color = options.filteredColor, \
                     label = 'filtered series')

            if options.showConfidenceInterval:
                upper, lower = getInterval(result.filteredObs[start : end], \
                                           result.filteredObsVar[start : end], \
                                           p = options.confidence)
                    
                plotInterval(time = time[start : end], \
                             upper = to1dArray(upper), lower = to1dArray(lower), \
                             color = options.filteredColor)
        plt.legend(loc='best', shadow = True) #, fontsize = 'x-large')
        location += 1

    # plot predicted results if needed
    if options.plotPredictedData:
        # the location
        subplot(size, location)
        
        #plot original data
        plotData(time = time, data = data, showDataPoint = \
                 options.showDataPoint, color = options.dataColor, label = 'time series')
        
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        if start < end:
            plotData(time = time[start : end], \
                     data = to1dArray(result.predictedObs), \
                     showDataPoint = options.showFittedPoint, \
                     color = options.predictedColor, \
                     label = 'one-day prediction')

            if options.showConfidenceInterval:
                upper, lower = getInterval(result.predictedObs[start : end], \
                                           result.filteredObsVar[start : end], \
                                           p = options.confidence)
                plotInterval(time = time[start:end], \
                             upper = to1dArray(upper), lower = to1dArray(lower), \
                             color = options.predictedColor)
        plt.legend(loc='best', shadow = True)    
        location += 1

    # plot smoothed results if needed
    if options.plotSmoothedData:
        # the location
        subplot(size, location)
        
        #plot original data
        plotData(time = time, data = data, showDataPoint = \
                 options.showDataPoint, color = options.dataColor, label = 'time series')
        
        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        if start < end:
            plotData(time = time[start:end], \
                     data = to1dArray(result.smoothedObs), \
                     showDataPoint = options.showFittedPoint, \
                     color = options.smoothedColor, \
                     label = 'smoothed series')
            
            if options.showConfidenceInterval:
                upper, lower = getInterval(result.smoothedObs[start : end], \
                                           result.smoothedObsVar[start : end], \
                                           p = options.confidence)
                plotInterval(time = time[start:end], \
                             upper = to1dArray(upper), lower = to1dArray(lower), \
                             color = options.smoothedColor)
        plt.legend(loc='best', shadow = True)
        
def plotData(time, data, showDataPoint = True, color = 'black', label = 'unknown'):
    """
    The function to plot data points.

    Args:
        time: time label
        data: the data points
        showDataPoint: indicate whether scatter plot is needed
        color: the color of the plot
        label: the label for the plot
    """
    if time is None:
        if showDataPoint:
            plt.plot(data, 'o', color = color)
            plt.plot(data, '-', color = color, label = label)
        else:
            plt.plot(data, '-', color = color, label = label)
    else:
        if showDataPoint:
            plt.plot(time, data, 'o', color = color)
            plt.plot(time, data, '-', color = color, label = label)
        else:
            plt.plot(time, data, '-', color = color, label = label)

 
def plotInterval(time, upper, lower, color = 'black'):
    """
    The function to plot confidence interval.

    Args:
        time: time label
        upper: the upper bound
        lower: the lower bound
        color: the color of the plot
    """
    if time is None:
        plt.plot(upper, '--', color = color)
        plt.plot(lower, '--', color = color)
    else:
        plt.plot(time, upper, '--', color = color)
        plt.plot(time, lower, '--', color = color)


def plotInitialize():
    """
    Initialize the plot

    """
    plt.figure()

 
def subplot(size, location):
    """
    Used for plotting multiple figures

    """
    plt.subplot(str(size[0]) + str(size[1]) + str(location))


def plotout():
    """
    Show the plot

    """
    plt.show()


def to1dArray(arrayOf1dMatrix):
    """
    Convert numpy style matrix to usual array

    """
    return [item.tolist()[0][0] for item in arrayOf1dMatrix]
