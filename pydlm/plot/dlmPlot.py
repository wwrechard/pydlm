"""
==================================================================================

Code for plot dlm results

==================================================================================

This module provide the plotting functionality for dlm

"""
import matplotlib.pyplot as plt
from pydlm.base.tools import getInterval

# =========================== plot for main data ============================


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
    plotData(time=time, data=data,
             showDataPoint=options.showDataPoint, color=options.dataColor,
             label='time series')

    # plot fitered results if needed
    if options.plotFilteredData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time=time[start:end],
                 data=to1dArray(result.filteredObs[start:end]),
                 showDataPoint=options.showFittedPoint,
                 color=options.filteredColor,
                 label='filtered series')

        if options.showConfidenceInterval:
            upper, lower = getInterval(result.filteredObs[start:end],
                                       result.filteredObsVar[start:end],
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=to1dArray(upper), lower=to1dArray(lower),
                         intervalType=options.intervalType,
                         color=options.filteredColor)

        # plot predicted results if needed
    if options.plotPredictedData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time=time[start:end],
                 data=to1dArray(result.predictedObs),
                 showDataPoint=options.showFittedPoint,
                 color=options.predictedColor,
                 label='one-day prediction')

        if options.showConfidenceInterval:
            upper, lower = getInterval(result.predictedObs[start:end],
                                       result.filteredObsVar[start:end],
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=to1dArray(upper), lower=to1dArray(lower),
                         intervalType=options.intervalType,
                         color=options.predictedColor)

    # plot smoothed results if needed
    if options.plotSmoothedData:
        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        plotData(time=time[start:end],
                 data=to1dArray(result.smoothedObs),
                 showDataPoint=options.showFittedPoint,
                 color=options.smoothedColor,
                 label='smoothed series')

        if options.showConfidenceInterval:
            upper, lower = getInterval(result.smoothedObs[start:end],
                                       result.smoothedObsVar[start:end],
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=to1dArray(upper), lower=to1dArray(lower),
                         intervalType=options.intervalType,
                         color=options.smoothedColor)

    plt.legend(loc='best', shadow=True)  # , fontsize = 'x-large')


def plotInMultipleFigure(time, data, result, options):
    """
    Plot the dlm results in multiple figure, each with one result and the
    original data

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

        # plot original data
        plotData(time=time, data=data,
                 showDataPoint=options.showDataPoint, color=options.dataColor,
                 label='time series')

        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        if start < end:
            plotData(time=time[start:end],
                     data=to1dArray(result.filteredObs[start:end]),
                     showDataPoint=options.showFittedPoint,
                     color=options.filteredColor,
                     label='filtered series')

            if options.showConfidenceInterval:
                upper, lower = getInterval(result.filteredObs[start:end],
                                           result.filteredObsVar[start:end],
                                           p=options.confidence)

                plotInterval(time=time[start:end],
                             upper=to1dArray(upper), lower=to1dArray(lower),
                             intervalType=options.intervalType,
                             color=options.filteredColor)
        plt.legend(loc='best', shadow=True)  # , fontsize = 'x-large')
        location += 1

    # plot predicted results if needed
    if options.plotPredictedData:
        # the location
        subplot(size, location)

        # plot original data
        plotData(time=time, data=data,
                 showDataPoint=options.showDataPoint,
                 color=options.dataColor, label='time series')

        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        if start < end:
            plotData(time=time[start:end],
                     data=to1dArray(result.predictedObs),
                     showDataPoint=options.showFittedPoint,
                     color=options.predictedColor,
                     label='one-day prediction')

            if options.showConfidenceInterval:
                upper, lower = getInterval(result.predictedObs[start:end],
                                           result.filteredObsVar[start:end],
                                           p=options.confidence)
                plotInterval(time=time[start:end],
                             upper=to1dArray(upper), lower=to1dArray(lower),
                             intervalType=options.intervalType,
                             color=options.predictedColor)
        plt.legend(loc='best', shadow=True)
        location += 1

    # plot smoothed results if needed
    if options.plotSmoothedData:
        # the location
        subplot(size, location)

        # plot original data
        plotData(time=time, data=data,
                 showDataPoint=options.showDataPoint,
                 color=options.dataColor, label='time series')

        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        if start < end:
            plotData(time=time[start:end],
                     data=to1dArray(result.smoothedObs),
                     showDataPoint=options.showFittedPoint,
                     color=options.smoothedColor,
                     label='smoothed series')

            if options.showConfidenceInterval:
                upper, lower = getInterval(result.smoothedObs[start:end],
                                           result.smoothedObsVar[start:end],
                                           p=options.confidence)
                plotInterval(time=time[start:end],
                             upper=to1dArray(upper), lower=to1dArray(lower),
                             intervalType=options.intervalType,
                             color=options.smoothedColor)
        plt.legend(loc='best', shadow=True)

# ============================ plot for latents ============================


def plotLatentState(time, coordinates, result, options, name):
    """
    Plot the latent state for given coordinates

    Args:
        time: the time label
        coordinates: the coordinates to plot
        result: the fitted result from dlm class
        options: options for the plot, for details please refer to @dlm
        name: the name of the component
    """
    # first compute how many plots are needed
    numOfPlots = len(coordinates)
    size = (numOfPlots, 1)

    # plot all needed results
    for i, dim in enumerate(coordinates):
        subplot(size, i + 1)
        plotSingleState(time, dim, result, options)
        plt.title('Filter result for dimension ' + str(i) +
                  ' in component: ' + name)


def plotSingleState(time, dimension, result, options):
    """
    Plot a single coordinate in the latent state vector

    Args:
        time: the time label
        dimension: the coordinate of the plot
        result: the fitted result from dlm class
        options: options for the plot, for details please refer to @dlm
    """

    # plot fitered results if needed
    if options.plotFilteredData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        data = [item[dimension, 0] for item in result.filteredState[start:end]]
        var = [abs(item[dimension, dimension])
               for item in result.filteredCov[start:end]]

        plotData(time=time[start:end],
                 data=data,
                 showDataPoint=options.showFittedPoint,
                 color=options.filteredColor,
                 label='filtered state')

        if options.showConfidenceInterval:
            upper, lower = getInterval(data, var, p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.filteredColor)

    # plot predicted results if needed
    if options.plotPredictedData:
        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        data = [item[dimension, 0]
                for item in result.predictedState[start:end]]
        var = [abs(item[dimension, dimension])
               for item in result.predictedCov[start:end]]

        plotData(time=time[start:end],
                 data=data,
                 showDataPoint=options.showFittedPoint,
                 color=options.predictedColor,
                 label='predicted state')

        if options.showConfidenceInterval:
            upper, lower = getInterval(data, var, p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.predictedColor)

    # plot smoothed results if needed
    if options.plotSmoothedData:
        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        data = [item[dimension, 0] for item in result.smoothedState[start:end]]
        var = [abs(item[dimension, dimension])
               for item in result.smoothedCov[start:end]]

        plotData(time=time[start:end],
                 data=data,
                 showDataPoint=options.showFittedPoint,
                 color=options.smoothedColor,
                 label='smoothed state')

        if options.showConfidenceInterval:
            upper, lower = getInterval(data, var, p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.smoothedColor)

    plt.legend(loc='best', shadow=True)

# ============================ plot for component =============================


def plotComponent(time, data, result, options):
    """
    Plot the results of a single component in one figure

    Args:
        time: the time label
        data: a dictionary contains all information. The basic keys are
              data['filteredMean'] = ...
              data['filteredVar']  = ...
              data['predictedMean']= ...
              data['predictedVar'] = ...
              data['smoothedMean'] = ...
              data['smoothedVar']  = ...
              data['name']         = ...
              not all are needed, depending on the setting in options.
        result: the fitted result from dlm class
        options: options for the plot, for details please refer to @dlm
    """

    if options.separatePlot:
        numOfPlots = options.plotFilteredData + options.plotPredictedData + \
                     options.plotSmoothedData
        size = (numOfPlots, 1)
        location = 1

    # plot fitered results if needed
    if options.plotFilteredData:
        if options.separatePlot:
            subplot(size, location)
            location += 1

        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time=time[start:end],
                 data=data['filteredMean'],
                 showDataPoint=options.showFittedPoint,
                 color=options.filteredColor,
                 label='filtered ' + data['name'])

        if options.showConfidenceInterval:
            upper, lower = getInterval(data['filteredMean'],
                                       list(map(abs, data['filteredVar'])),
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.filteredColor)

        plt.legend(loc='best', shadow=True)

    # plot predicted results if needed
    if options.plotPredictedData:
        if options.separatePlot:
            subplot(size, location)
            location += 1

        start = result.filteredSteps[0]
        end = result.filteredSteps[1] + 1
        plotData(time=time[start:end],
                 data=data['predictedMean'],
                 showDataPoint=options.showFittedPoint,
                 color=options.predictedColor,
                 label='one-step predict for ' + data['name'])

        if options.showConfidenceInterval:
            upper, lower = getInterval(data['predictedMean'],
                                       list(map(abs, data['predictedVar'])),
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.predictedColor)

        plt.legend(loc='best', shadow=True)

    # plot smoothed results if needed
    if options.plotSmoothedData:
        if options.separatePlot:
            subplot(size, location)
            location += 1

        start = result.smoothedSteps[0]
        end = result.smoothedSteps[1] + 1
        plotData(time=time[start:end],
                 data=data['smoothedMean'],
                 showDataPoint=options.showFittedPoint,
                 color=options.smoothedColor,
                 label='smoothed ' + data['name'])

        if options.showConfidenceInterval:
            upper, lower = getInterval(data['smoothedMean'],
                                       list(map(abs, data['smoothedVar'])),
                                       p=options.confidence)

            plotInterval(time=time[start:end],
                         upper=upper, lower=lower,
                         intervalType=options.intervalType,
                         color=options.smoothedColor)

        plt.legend(loc='best', shadow=True)


# =========================== plot for prediction ========================
def plotPrediction(time, data, predictedTime,
                   predictedData, predictedVar, options):
    """
    Function for ploting N-day ahead prediction

    Args:
        time: the data time
        data: the original data
        predictedTime: the predicted time period
        predictedData: the predicted data
        predictedVar: the predicted variance
        options: options for the plot, for details please refer to @dlm
    """
    # plot the original data
    plotData(time, data, showDataPoint=options.showDataPoint,
             color=options.dataColor,
             label='time series')
    # overlay the predicted data
    plotData(predictedTime, predictedData,
             showDataPoint=options.showFittedPoint,
             color=options.predictedColor,
             label='N-day prediction')
    # overlay confidence interval
    if options.showConfidenceInterval:
        upper, lower = getInterval(predictedData,
                                   predictedVar,
                                   p=options.confidence)

        plotInterval(time=predictedTime,
                     upper=upper, lower=lower,
                     intervalType=options.intervalType,
                     color=options.predictedColor)
    
    
# =========================== basic functions ============================


def plotData(time, data, showDataPoint=True, color='black', label='unknown'):
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
            plt.plot(data, 'o', color=color)
            plt.plot(data, '-', color=color, label=label)
        else:
            plt.plot(data, '-', color=color, label=label)
    else:
        if showDataPoint:
            plt.plot(time, data, 'o', color=color)
            plt.plot(time, data, '-', color=color, label=label)
        else:
            plt.plot(time, data, '-', color=color, label=label)


def plotInterval(time, upper, lower, intervalType, color='black'):
    """
    The function to plot confidence interval.

    Args:
        time: time label
        upper: the upper bound
        lower: the lower bound
        color: the color of the plot
    """
    ALPHA = 0.4
    if time is None:
        if intervalType == 'line':
            plt.plot(upper, '--', color=color)
            plt.plot(lower, '--', color=color)
        elif intervalType == 'ribbon':
            plt.fill_between(upper, lower, facecolor=color,
                             alpha=ALPHA)
    else:
        if intervalType == 'line':
            plt.plot(time, upper, '--', color=color)
            plt.plot(time, lower, '--', color=color)
        elif intervalType == 'ribbon':
            plt.fill_between(time, upper, lower,
                             facecolor=color, alpha=ALPHA)


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
