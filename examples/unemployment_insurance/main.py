#==================================================================
#
#  Put data description here
#
#==================================================================

# Read data file
import os
this_dir = os.getcwd()
DATA_PATH = os.path.join(this_dir, "data.csv")
data_file = open(DATA_PATH, 'r')

variables = data_file.readline().strip().split(',')
data_map = {}
for var in variables:
    data_map[var] = []

for line in data_file:
    for i, data_piece in enumerate(line.strip().split(',')):
        data_map[variables[i]].append(float(data_piece))

# plot the raw data
time_series = data_map[variables[0]]

import matplotlib.pyplot as plt
import pydlm.plot.dlmPlot as dlmPlot
dlmPlot.plotData(range(len(time_series)),
                 time_series,
                 showDataPoint=False,
                 label='raw_data')
plt.legend(loc='best', shadow=True)
plt.show()

# Build a simple model
from pydlm import dlm, trend, seasonality

# A linear trend
linear_trend = trend(degree=1, discount=0.95, name='linear_trend', w=10)
# A seasonality
seasonal52 = seasonality(period=52, discount=0.99, name='seasonal52', w=10)
# Build DLM and fit
simple_dlm = dlm(time_series) + linear_trend + seasonal52
simple_dlm.fit()
# Plot the fitted results
simple_dlm.plot()
# Plot each component (attribution)
simple_dlm.plot('linear_trend')
simple_dlm.plot('seasonal52')
