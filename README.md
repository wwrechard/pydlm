=======================================================
[PyDLM](https://pydlm.github.io/)
=======================================================

Welcome to [pydlm](https://pydlm.github.io/), a flexible, user-friendly and rich functionality time series modeling library for python. This package implementes the Bayesian dynamic linear model (Harrison and West, 1999) for time series data. Time series modeling is easy with `pydlm`. Complex models can be constructed via simple operations
```
  >>> #import dlm and its modeling components
  >>> from pydlm import dlm, trend, seasonality, dynamic, autoReg
  >>>
  >>> #randomly generate data
  >>> data = [0] * 100 + [3] * 100
  >>>
  >>> #construct the base
  >>> myDLM = dlm(data)  
  >>>
  >>> #adding model components
  >>> myDLM = myDLM + trend(2, name = 'lineTrend') #add a second-order trend (linear trending)
  >>> myDLM = myDLM + seasonality(7, name = 'day7') #add a 7 day seasonality
  >>> myDLM = myDLM + autoReg(3, data = data, name = 'ar3') #add a 3 step auto regression
  >>>
  >>> #show the added components
  >>> myDLM.ls()
  >>>
  >>> #delete unwanted component
  >>> myDLM.delete('ar3')
  >>> myDLM.ls()
```
Users can then analyze the data with the constructed model
```
  >>> #fit forward filter
  >>> myDLM.fitForwardFilter()
  >>>
  >>> #fit backward smoother
  >>> myDLM.fitBackwardSmoother()
  >>>
  >>> # get the filtered and smoothed results
  >>> filteredObs = myDLM.getFilteredObs()
  >>> smoothedObs = myDLM.getSmoothedObs()
```

and plot the results easily
<img align="right" src="/doc/source/img/readmePlot1.png" width="420"/>
```
  >>> #plot the results
  >>> myDLM.plot()
```
```
  >>> #plot only the filtered results
  >>> myDLM.turnOff('smoothed plot')
  >>> myDLM.plot()
```
```
  >>> #plot in one figure
  >>> myDLM.turnOff('multiple plots')
  >>> myDLM.plot()
```
<p align="center">
<img src="/doc/source/img/readmePlot2.png" width="430"/>
<img src="/doc/source/img/readmePlot3.png" width="422"/>
</p>
If users are unsatisfied with the model results, they can simply reconstruct the model and refit
```
  >>> myDLM = myDLM + seasonality(4)
  >>> myDLM.ls()
  >>> myDLM.fit()
```
`pydlm` supports missing observations
```
  >>> data = [1, 0, 0, 1, 0, 0, None, 0, 1, None, None, 0, 0]
  >>> myDLM = dlm(data) + trend(2)
  >>> myDLM.fit() #fit() will fit both forward filter and backward smoother
```
It also includes the discounting factor, which can be used to control how rapid the model should adapt to the new data (More details will be provided in the documentation)
```
  >>> data = [0] * 100 + [3] * 100
  >>> myDLM = dlm(data) + trend(2, discount = 1.0)
  >>> myDLM.fit()
  >>> myDLM.plot()
  >>>
  >>> myDLM.delete('trend')
  >>> myDLM = myDLM + trend(2, discount = 0.8)
  >>> myDLM.fit()
  >>> myDLM.plot()
```
<p align="center">
<img src="/doc/source/img/readmePlot4.png" width="416"/>
<img src="/doc/source/img/readmePlot5.png" width="446"/>
</p>
For online updates
```
  >>> myDLM = dlm([]) + trend(2) + seasonality(7)
  >>> for t in range(0, len(data)):
  ...     myDLM.append([data[t]])
  ...     myDLM.fitForwardFilter()
  >>> filteredObs = myDLM.getFilteredObs()
```  
------------
Installation
------------
You can currently get the package from `pypi` by

      $ pip install pydlm

You can also get the latest from [github]
(https://github.com/wwrechard/PyDLM)

      $ git clone git@github.com:wwrechard/pydlm.git pydlm
      $ cd pydlm
      $ sudo python setup.py install

`pydlm` depends on the following modules,

* `numpy`     (for core functionality)
*  `matplotlib` (for plotting results)
* `Sphinx`    (for generating documentation)
* `unittest`  (for testing)

-------------
Documentation
-------------
Detailed documentation is provided in [PyDLM](https://pydlm.github.io/) with special attention to the [User manual](https://pydlm.github.io/#dynamic-linear-models-user-manual).
