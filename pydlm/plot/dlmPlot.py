import matplotlib.pyplot as plt

class dlmPlot:

    @staticmethod
    def plotData(time, data, showDataPoint = True, color = 'black'):
        if time is None:
            if showDataPoint:
                plt.plot(data, 'o', color = color)
                plt.plot(data, '-', color = color)
            else:
                plt.plot(data, '-', color = color)
        else:
            if showDataPoint:
                plt.plot(time, data, 'o', color = color)
                plt.plot(time, data, '-', color = color)
            else:
                plt.plot(time, data, '-', color = color)

    @staticmethod
    def plotInterval(time, upper, lower, color = 'black'):
        if time is None:
            plt.plot(upper, '--', color = color)
            plt.plot(lower, '--', color = color)
        else:
            plt.plot(time, upper, '--', color = color)
            plt.plot(time, lower, '--', color = color)

    @staticmethod
    def plotInitialize():
        plt.figure(1)

    @staticmethod
    def subplot(size, location):
        plt.subplot(str(size[0]) + str(size[1]) + str(location))

    @staticmethod
    def plotout():
        plt.show()
